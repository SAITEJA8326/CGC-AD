"""Training, evaluation, and the deployment-stream simulator.

The expensive work happens exactly once: `build_banks` extracts frozen-backbone
features for the normals and for every rung of the hardness ladder, caches them
to disk, and every subsequent arm of the sweep reuses that cache. Training,
every controller, every threshold policy, and every seed then run on cached
tensors and cost seconds.

Both long loops (`train_head`, `run_stream`) checkpoint periodically and resume
exactly, including RNG state, so a killed Kaggle session costs at most
`ckpt.every_steps` steps of work.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import synth as synth_mod
from .checkpoint import (CheckpointManager, banks_path, disk_free_gb,
                         load_banks, rng_state, save_banks, set_rng_state)
from .conformal import (BaseThreshold, PowerMartingale, build_threshold,
                        controlled_quantity,
                        conformal_pvalue, conformal_quantile)
from .curriculum import BaseController, build_controller, miss_rate
from .data import ClassData, apply_drift, severity_at
from .model import FrozenBackbone, PatchHead, image_score


# --------------------------------------------------------------------------
# Feature extraction and banks
# --------------------------------------------------------------------------

@torch.no_grad()
def extract(backbone: FrozenBackbone, images: torch.Tensor, device: str,
            batch_size: int = 32, amp: bool = True) -> torch.Tensor:
    """(N,3,H,W) in [0,1] -> (N,D,S,S) fp16 on CPU."""
    out = []
    backbone.eval()
    for i in range(0, images.shape[0], batch_size):
        x = images[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                            enabled=amp and "cuda" in device):
            f = backbone(x)
        out.append(f.half().cpu())
    return torch.cat(out)


@dataclass
class Banks:
    """Cached features. `synth[k]` corresponds to hardness `levels[k]`."""
    normal: torch.Tensor                       # (N,D,S,S) fp16
    synth: List[torch.Tensor]                  # K x (N*V,D,S,S) fp16
    synth_masks: List[torch.Tensor]            # K x (N*V,1,S,S) fp16
    levels: np.ndarray
    calib: torch.Tensor                        # (C,D,S,S) fp16
    val_normal: torch.Tensor                   # (V,D,S,S) fp16
    val_synth: List[torch.Tensor]              # K x (V,D,S,S) fp16

    def nbytes_gb(self) -> float:
        t = [self.normal, self.calib, self.val_normal]
        t += self.synth + self.synth_masks + self.val_synth
        return sum(x.numel() * x.element_size() for x in t) / 1e9


def build_banks(backbone: FrozenBackbone, cd: ClassData, cfg,
                rng: np.random.Generator, device: str,
                tracker=None, cls: Optional[str] = None) -> Banks:
    """Extract (or load from cache) every feature bank for one class."""
    log = tracker.logger if tracker else None
    path = banks_path(cfg, cls or cd.name) if cfg.ckpt.cache_banks else None

    if path and cfg.ckpt.resume:
        banks = load_banks(path, log)
        if banks is not None:
            return banks

    sc, mc = cfg.synth, cfg.model
    levels = synth_mod.hardness_ladder(sc.n_levels)
    amp, bs = cfg.train.amp, cfg.train.batch_size
    t0 = time.time()

    normal = extract(backbone, cd.train, device, bs, amp)
    calib = extract(backbone, cd.calib, device, bs, amp)
    val_normal = extract(backbone, cd.val, device, bs, amp)

    synth, masks, val_synth = [], [], []
    for li, h in enumerate(levels):
        xs, ms = [], []
        for _ in range(sc.variants_per_image):
            x, m = synth_mod.synth_batch(cd.train, float(h), sc, rng)
            xs.append(x)
            ms.append(m)
        x = torch.cat(xs)
        m = torch.cat(ms)
        synth.append(extract(backbone, x, device, bs, amp))
        masks.append(F.interpolate(m, size=(mc.feat_size, mc.feat_size),
                                   mode="area").half())
        vx, _ = synth_mod.synth_batch(cd.val, float(h), sc, rng)
        val_synth.append(extract(backbone, vx, device, bs, amp))
        if tracker:
            tracker.info(f"  bank level {li + 1}/{len(levels)} (h={h:.2f}) done")

    banks = Banks(normal, synth, masks, levels, calib, val_normal, val_synth)
    if tracker:
        tracker.info(f"banks built in {time.time() - t0:.1f}s, "
                     f"{banks.nbytes_gb():.2f} GB in RAM")
        tracker.scalars(0, phase="banks", size_gb=banks.nbytes_gb(),
                        build_seconds=round(time.time() - t0, 1),
                        **tracker.gpu_mem())

    if path:
        free = disk_free_gb(os.path.dirname(path))
        if free - banks.nbytes_gb() > cfg.ckpt.min_free_gb:
            save_banks(banks, path, log)
        elif log:
            log.warning(f"skipping bank cache: only {free:.1f} GB free on "
                        f"{os.path.dirname(path)}")
    return banks


def _nearest_level(banks: Banks, h: float) -> int:
    return int(np.argmin(np.abs(banks.levels - float(h))))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@torch.no_grad()
def score_features(head: PatchHead, feats: torch.Tensor, device: str,
                   topk_frac: float, batch_size: int = 128
                   ) -> Tuple[np.ndarray, torch.Tensor]:
    head.eval()
    scores, maps = [], []
    for i in range(0, feats.shape[0], batch_size):
        f = feats[i:i + batch_size].to(device).float()
        pl = head(f)
        scores.append(image_score(pl, topk_frac).cpu().numpy())
        maps.append(pl.cpu())
    return np.concatenate(scores), torch.cat(maps)


def _val_auroc(head: PatchHead, banks: Banks, cfg, device: str) -> float:
    """Held-out separability of normals vs synthetic anomalies (model selection).

    Uses the validation split only -- never the calibration split, and never
    the test split. See the leakage note in the review (F11).
    """
    n, _ = score_features(head, banks.val_normal, device, cfg.model.topk_frac)
    s = np.concatenate([
        score_features(head, banks.val_synth[k], device, cfg.model.topk_frac)[0]
        for k in range(len(banks.levels))])
    y = np.r_[np.zeros(len(n)), np.ones(len(s))]
    return float(roc_auc_score(y, np.r_[n, s]))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_head(banks: Banks, cfg, controller: BaseController, device: str,
               rng: np.random.Generator, tracker=None,
               ckpt: Optional[CheckpointManager] = None
               ) -> Tuple[PatchHead, Dict]:
    """Train the patch head with a curriculum over the hardness ladder.

    Every arm is matched on gradient steps (`train.max_steps`) and on synthetic
    budget (one synthetic batch per normal batch), so a win cannot be explained
    by extra compute.

    Resumable: if `ckpt` holds a `latest` checkpoint and `cfg.ckpt.resume` is
    set, weights, optimiser, scaler, controller, step counter, and all RNG
    streams are restored and training continues from where it stopped.
    """
    head = PatchHead(banks.normal.shape[1], cfg.model.head_width).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and "cuda" in device)
    bce = nn.BCEWithLogitsLoss()

    N = banks.normal.shape[0]
    bs = min(cfg.train.batch_size, N)
    steps_per_epoch = max(1, N // bs)
    total = cfg.train.max_steps or cfg.train.epochs * steps_per_epoch
    step, epoch, t0 = 0, 0, time.time()
    resumed_from = None

    if ckpt is not None and cfg.ckpt.resume:
        st = ckpt.load_latest()
        if st is not None:
            head.load_state_dict(st["head"])
            opt.load_state_dict(st["opt"])
            if st.get("scaler") is not None:
                scaler.load_state_dict(st["scaler"])
            controller.load_state_dict(st["controller"])
            set_rng_state(st.get("rng"), rng)
            step, epoch = int(st["step"]), int(st.get("epoch", 0))
            resumed_from = step
            if tracker:
                tracker.info(f"resumed training from step {step}/{total}")

    while step < total:
        head.train()
        for _ in range(steps_per_epoch):
            if step >= total:
                break
            ni = rng.integers(0, N, size=bs)
            k = _nearest_level(banks, controller.sample_hardness())
            si = rng.integers(0, banks.synth[k].shape[0], size=bs)

            fn = banks.normal[ni].to(device).float()
            fs = banks.synth[k][si].to(device).float()
            ms = banks.synth_masks[k][si].to(device).float()

            f = torch.cat([fn, fs])
            y = torch.cat([torch.zeros_like(ms), ms])
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                enabled=cfg.train.amp and "cuda" in device):
                loss = bce(head(f), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1

            if tracker and step % max(1, cfg.logging.log_every) == 0:
                tracker.scalars(step, phase="train", loss=float(loss.item()),
                                h=controller.hardness(),
                                lr=opt.param_groups[0]["lr"])

        # ---- controller feedback, computed on held-out data only ----------
        cal_s, _ = score_features(head, banks.calib, device, cfg.model.topk_frac)
        per_level = np.array([
            miss_rate(score_features(head, banks.val_synth[k], device,
                                     cfg.model.topk_frac)[0],
                      cal_s, cfg.curriculum, cfg.threshold.alpha)
            for k in range(len(banks.levels))])
        k_cur = _nearest_level(banks, controller.hardness())
        controller.update(float(per_level[k_cur]), per_level)
        epoch += 1

        vauc = _val_auroc(head, banks, cfg, device)
        if tracker:
            tracker.scalars(step, phase="epoch", epoch=epoch, val_auroc=vauc,
                            h=controller.hardness(),
                            miss_at_h=float(per_level[k_cur]),
                            calib_tau=float(conformal_quantile(
                                cal_s, cfg.threshold.alpha)),
                            **tracker.gpu_mem())

        if ckpt is not None:
            state = {"head": head.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                     "controller": controller.state_dict(), "epoch": epoch,
                     "rng": rng_state(rng), "val_auroc": vauc}
            if step % max(1, cfg.ckpt.every_steps) < steps_per_epoch or step >= total:
                ckpt.save(state, step)
            if cfg.ckpt.save_best:
                ckpt.save_best({"head": head.state_dict(),
                                "controller": controller.state_dict()},
                               vauc, step, higher_better=True)

    # Final weights come from `best` when best-tracking is on: model selection
    # runs on the validation split, the only split allowed to choose it.
    if ckpt is not None and cfg.ckpt.save_best:
        best = ckpt.load_best()
        if best is not None:
            head.load_state_dict(best["head"])
            if tracker:
                tracker.info(f"restored best head "
                             f"(val_auroc={best.get('metric', float('nan')):.4f})")

    return head, {"steps": step, "epochs": epoch, "resumed_from": resumed_from,
                  "train_seconds": round(time.time() - t0, 1)}


def pick_oracle_hardness(banks: Banks, cfg, device: str,
                         rng: np.random.Generator, tracker=None) -> float:
    """Grid search fixed hardness on the validation split. Upper-bound control."""
    from .config import CurriculumConfig
    best_h, best_auc = float(banks.levels[0]), -1.0
    for h in banks.levels:
        c = CurriculumConfig(**{**cfg.curriculum.__dict__, "kind": "fixed",
                                "h0": float(h)})
        sub = _shallow_copy_cfg(cfg, curriculum=c)
        sub.train.max_steps = max(50, (cfg.train.max_steps or 400) // 4)
        sub.ckpt.enabled = False
        sub.ckpt.save_best = False
        head, _ = train_head(banks, sub,
                             build_controller(c, rng, len(banks.levels)),
                             device, rng)
        auc = _val_auroc(head, banks, sub, device)
        if tracker:
            tracker.info(f"  oracle probe h={h:.2f} val_auroc={auc:.4f}")
        if auc > best_auc:
            best_auc, best_h = auc, float(h)
    if tracker:
        tracker.info(f"oracle hardness = {best_h:.2f} (val_auroc={best_auc:.4f})")
    return best_h


def _shallow_copy_cfg(cfg, **kw):
    import copy
    c = copy.deepcopy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --------------------------------------------------------------------------
# Phase-1 gate: is hardness monotone in detectability?
# --------------------------------------------------------------------------

def sweep_monotonicity(head: PatchHead, banks: Banks, cfg, device: str,
                       tracker=None) -> Dict[str, object]:
    """Miss-rate as a function of hardness, on held-out data.

    If this is non-monotone or flat, a proportional setpoint controller is
    chasing an ill-posed target and you must redefine the hardness axis or
    switch to the distributional controller. Run this BEFORE anything else.
    """
    cal, _ = score_features(head, banks.calib, device, cfg.model.topk_frac)
    miss = []
    for k in range(len(banks.levels)):
        s, _ = score_features(head, banks.val_synth[k], device, cfg.model.topk_frac)
        miss.append(miss_rate(s, cal, cfg.curriculum, cfg.threshold.alpha))
    miss = np.array(miss)
    d = np.diff(miss)
    out = {
        "levels": banks.levels.tolist(),
        "miss_rate": miss.tolist(),
        "monotone_increasing": bool(np.all(d >= -1e-3)),
        "spearman": float(_spearman(banks.levels, miss)),
        "range": float(miss.max() - miss.min()),
    }
    out["gate_pass"] = bool(out["spearman"] > 0.8 and out["range"] > 0.3)
    if tracker:
        tracker.artifact("monotonicity", out)
        tracker.info(
            f"monotonicity gate: spearman={out['spearman']:.3f} "
            f"range={out['range']:.3f} -> "
            + ("PASS" if out["gate_pass"] else
               "FAIL: redefine the hardness axis or use "
               "curriculum.kind=distributional"))
    return out


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = (np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / d) if d > 0 else 0.0


# --------------------------------------------------------------------------
# Static evaluation
# --------------------------------------------------------------------------

def evaluate_static(head: PatchHead, backbone: FrozenBackbone, cd: ClassData,
                    cfg, device: str, drift_kind: str = "none",
                    severity: float = 0.0) -> Dict[str, float]:
    x = apply_drift(cd.test_images, drift_kind, severity)
    f = extract(backbone, x, device, cfg.train.batch_size, cfg.train.amp)
    s, maps = score_features(head, f, device, cfg.model.topk_frac)
    y = cd.test_labels.numpy()
    out = {"image_auroc": float(roc_auc_score(y, s)) if y.max() > 0 else float("nan")}

    m = F.interpolate(maps.float(), size=cd.test_masks.shape[-2:],
                      mode="bilinear", align_corners=False)
    gt = cd.test_masks.numpy().ravel().astype(int)
    if gt.max() > 0:
        pr = m.numpy().ravel()
        idx = np.random.default_rng(0).choice(gt.size, size=min(400_000, gt.size),
                                              replace=False)
        out["pixel_auroc"] = float(roc_auc_score(gt[idx], pr[idx]))
    return out


# --------------------------------------------------------------------------
# Deployment stream
# --------------------------------------------------------------------------

def _windowed(x: np.ndarray, w: int) -> np.ndarray:
    if x.size == 0:
        return x
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(x.size)
    lo = np.maximum(0, idx - w + 1)
    return (c[idx + 1] - c[lo]) / (idx - lo + 1)


def stream_metrics(records: List[Dict], alpha: float, window: int,
                   onset: int) -> Dict[str, float]:
    """FPR-centric metrics. AUROC cannot move in response to a thresholder;
    these can. See review F3."""
    lbl = np.array([r["label"] for r in records])
    flg = np.array([r["flag"] for r in records], dtype=float)
    scr = np.array([r["score"] for r in records], dtype=float)
    post = np.arange(len(records)) >= onset

    norm = lbl == 0
    fp = (flg == 1) & norm
    fpr_trace = _windowed(fp[norm].astype(float), window)
    dev = np.abs(fpr_trace - alpha)

    norm_post = norm & post
    fpr_post = float(fp[norm_post].mean()) if norm_post.any() else float("nan")
    anom_post = (lbl == 1) & post
    tpr_post = float(flg[anom_post].mean()) if anom_post.any() else float("nan")

    band = (fpr_trace >= 0.5 * alpha) & (fpr_trace <= 1.5 * alpha)
    n_onset = int(norm[:onset].sum())
    delay = float("nan")
    if n_onset < band.size:
        after = np.where(band[n_onset:])[0]
        delay = float(after[0]) if after.size else float("nan")

    out = {
        "mad_fpr": float(np.nanmean(dev[n_onset:])) if n_onset < dev.size else float("nan"),
        "mad_fpr_all": float(np.nanmean(dev)),
        "max_fpr_dev": float(np.nanmax(dev[n_onset:])) if n_onset < dev.size else float("nan"),
        "fpr_post": fpr_post,
        "tpr_post": tpr_post,
        "n_false_alarms": int(fp.sum()),
        "n_missed": int(((flg == 0) & (lbl == 1)).sum()),
        "recovery_delay": delay,
        # alpha_t under ACI is a sawtooth (one large downswing per false alarm,
        # slow recovery), so the endpoint value is meaningless. Report the
        # time-average of tau and alpha_t instead.
        "mean_alpha_t": float(np.mean([r["alpha_t"] for r in records])),
        "mean_tau_post": float(np.mean([r["tau"] for r in records if r["t"] >= onset])),
        "mean_tau_pre": float(np.mean([r["tau"] for r in records if r["t"] < onset])),
    }
    if lbl.max() > 0 and lbl.min() == 0:
        out["stream_auroc"] = float(roc_auc_score(lbl, scr))
        if post.any() and lbl[post].max() > 0 and lbl[post].min() == 0:
            out["stream_auroc_post"] = float(roc_auc_score(lbl[post], scr[post]))
    return out


_TRACE_KEYS = ("t", "score", "flag", "label", "tau", "alpha_t",
               "severity", "admitted", "martingale", "h")


def save_trace(records: List[Dict], path: str) -> None:
    """Per-step trace for the tau / FPR figures. Written atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **{k: np.array([r[k] for r in records])
                                for k in _TRACE_KEYS})
    os.replace(tmp, path)


def run_stream(head: PatchHead, backbone: FrozenBackbone, cd: ClassData,
               banks: Banks, controller: BaseController, cfg, device: str,
               rng: np.random.Generator, tracker=None,
               ckpt: Optional[CheckpointManager] = None
               ) -> Tuple[Dict[str, float], List[Dict]]:
    """Simulate deployment: drift ramps in at `onset`, the thresholder adapts,
    and (optionally) the head is re-adapted on the drifted buffer.

    The `couple` flag is the RQ3 switch. Both arms get the same trigger, the
    same adaptation budget, and the same buffer; they differ only in whether
    the live calibration state chooses the hardness of the synthetic
    supervision used during adaptation.

    Resumable at `ckpt.stream_every` granularity.
    """
    sc = cfg.stream
    if cfg.threshold.policy == "static" and sc.adapt and tracker:
        tracker.warn("threshold.policy=static with stream.adapt=true: the head "
                     "is still fine-tuned on adaptation events, so this is not a "
                     "static baseline. Set stream.adapt=false for threshold studies.")
    norm_pool = torch.nonzero(cd.test_labels == 0).flatten().numpy()
    anom_pool = torch.nonzero(cd.test_labels == 1).flatten().numpy()
    if anom_pool.size == 0:
        raise ValueError(f"class {cd.name} has no anomalous test images")

    # Materialise the whole stream from its own dedicated RNG, then batch
    # extract. A resumed run therefore replays the identical stream regardless
    # of how far the training RNG advanced.
    srng = np.random.default_rng(cfg.seed * 7919 + 13)
    prev = np.full(sc.steps, sc.prevalence, dtype=float)
    if getattr(sc, "burst_start", -1) >= 0 and getattr(sc, "burst_len", 0) > 0:
        prev[sc.burst_start:sc.burst_start + sc.burst_len] = sc.burst_prevalence
    labels = (srng.random(sc.steps) < prev).astype(int)
    sev = np.array([severity_at(t, cfg.drift) for t in range(sc.steps)])
    idx = np.array([srng.choice(anom_pool if l else norm_pool) for l in labels])

    imgs = torch.stack([
        apply_drift(cd.test_images[i], cfg.drift.kind, float(s))
        for i, s in zip(idx, sev)])
    feats = extract(backbone, imgs, device, cfg.train.batch_size, cfg.train.amp)

    cal_scores, _ = score_features(head, banks.calib, device, cfg.model.topk_frac)
    cal_scores_initial = np.array(cal_scores, copy=True)
    thr = build_threshold(cfg.threshold, cal_scores)
    mart = PowerMartingale(lam=sc.adapt_lambda)

    img_buf: Deque[torch.Tensor] = deque(maxlen=cfg.threshold.window)
    records: List[Dict] = []
    n_adapt, last_adapt, t_start = 0, -10 ** 9, 0

    if ckpt is not None and cfg.ckpt.resume:
        st = ckpt.load_latest()
        if st is not None:
            head.load_state_dict(st["head"])
            thr.load_state_dict(st["thr"])
            mart.load_state_dict(st["mart"])
            controller.load_state_dict(st["controller"])
            set_rng_state(st.get("rng"), rng)
            records = st["records"]
            n_adapt, last_adapt = st["n_adapt"], st["last_adapt"]
            t_start = int(st["step"]) + 1
            if st.get("img_buf") is not None:
                img_buf = deque([x for x in st["img_buf"].float()],
                                maxlen=cfg.threshold.window)
            if tracker:
                tracker.info(f"resumed stream from t={t_start}/{sc.steps}")

    def _save(t: int) -> None:
        if ckpt is None:
            return
        buf = torch.stack(list(img_buf)).half() if len(img_buf) else None
        ckpt.save({"head": head.state_dict(), "thr": thr.state_dict(),
                   "mart": mart.state_dict(),
                   "controller": controller.state_dict(),
                   "records": records, "n_adapt": n_adapt,
                   "last_adapt": last_adapt, "img_buf": buf,
                   "rng": rng_state(rng)}, t)

    for t in range(t_start, sc.steps):
        f = feats[t:t + 1]
        s = float(score_features(head, f, device, cfg.model.topk_frac)[0][0])
        flag = thr.flag(s)
        label = int(labels[t])

        # ---- buffer admission -------------------------------------------
        # The protocol decides admission; `contamination` is then the
        # probability that a TRUE ANOMALY slips past the protocol anyway. It is
        # applied to every protocol, so eps means the same thing in each column
        # of the breakdown table. (Sweeping `prevalence` is a different
        # experiment and must not be conflated with this one.)
        if sc.buffer_protocol == "clean":
            admit = (label == 0)
        elif sc.buffer_protocol == "self_label":
            admit = not flag
        elif sc.buffer_protocol == "reviewed":
            # un-flagged parts are presumed nominal; flagged parts receive a
            # human disposition, so the nominal ones (false alarms) come back
            admit = (not flag) or (label == 0)
        elif sc.buffer_protocol == "gated":
            lo = np.quantile(np.fromiter(thr.buffer, dtype=np.float64),
                             sc.gate_percentile) if len(thr.buffer) >= 20 else np.inf
            admit = s <= lo
        else:
            raise ValueError(sc.buffer_protocol)
        if not admit and getattr(sc, "admit_explore", 0.0) > 0 \
                and rng.random() < sc.admit_explore:
            admit = True                       # exploration: break the ratchet
        if label == 1 and not admit and rng.random() < sc.contamination:
            admit = True

        thr.observe(s, admit, flagged=flag)
        if admit:
            img_buf.append(imgs[t].clone())

        p = conformal_pvalue(s, np.fromiter(thr.buffer, dtype=np.float64)) \
            if len(thr.buffer) >= 20 else 1.0
        alarm = mart.update(p) if admit else False

        records.append({"t": t, "score": s, "flag": int(flag), "label": label,
                        "tau": thr.tau, "alpha_t": thr.alpha_t,
                        "severity": float(sev[t]), "admitted": int(admit),
                        "martingale": mart.value, "h": controller.hardness()})

        # Buffer starvation is the leading indicator of a self-labelling
        # collapse: if admission dries up, the quantile estimate freezes on
        # stale data and every later metric is meaningless.
        if tracker and t == cfg.drift.onset + 199:
            ar = float(np.mean([r["admitted"] for r in records[-200:]]))
            if ar < 0.25:
                tracker.warn(f"buffer admission rate {ar:.2f} over the last 200 "
                             f"steps: the buffer is starving, treat downstream "
                             f"metrics for this run as invalid")

        # ---- online adaptation ------------------------------------------
        if sc.adapt and len(img_buf) >= 32 and (t - last_adapt) > 50:
            trig = ({"martingale": alarm,
                     "ema": abs(thr.alpha_t - cfg.threshold.alpha) > 0.02,
                     "timer": (t % max(1, sc.adapt_period) == 0 and t > 0),
                     "none": False}[sc.adapt_trigger])
            if trig:
                if tracker:
                    tracker.info(f"t={t}: adaptation event #{n_adapt + 1} "
                                 f"(couple={sc.couple}, sev={sev[t]:.2f})")
                head, controller = _adapt(head, backbone, list(img_buf), thr,
                                          controller, cfg, device, rng)
                cal_scores, _ = score_features(
                    head, banks.calib, device, cfg.model.topk_frac)
                buf_f = extract(backbone, torch.stack(list(img_buf)), device,
                                cfg.train.batch_size, cfg.train.amp)
                buf_s, _ = score_features(head, buf_f, device, cfg.model.topk_frac)
                # A static threshold is, by definition, never refit. Before
                # this guard, adaptation events rewrote tau for the "static"
                # arm too, so r3_static was really "static + martingale-
                # triggered recalibration" (1.24 events per run in E1).
                if thr.name != "static":
                    thr.tau = conformal_quantile(buf_s, thr.alpha_t)
                    thr.buffer = deque(buf_s.tolist(), maxlen=cfg.threshold.window)
                mart.reset()
                n_adapt += 1
                last_adapt = t
                if tracker:
                    tracker.scalars(t, phase="adapt", n_adapt=n_adapt,
                                    h=controller.hardness(), tau=thr.tau,
                                    alpha_t=thr.alpha_t)
                _save(t)

        if tracker and (t + 1) % max(1, cfg.logging.stream_log_every) == 0:
            recent = records[-cfg.logging.stream_log_every:]
            nz = [r for r in recent if r["label"] == 0]
            tracker.scalars(t, phase="stream", tau=thr.tau, alpha_t=thr.alpha_t,
                            severity=float(sev[t]),
                            fpr_window=float(np.mean([r["flag"] for r in nz]))
                            if nz else float("nan"),
                            martingale=mart.value, n_adapt=n_adapt)

        if ckpt is not None and (t + 1) % max(1, cfg.ckpt.stream_every) == 0:
            _save(t)

    m = stream_metrics(records, cfg.threshold.alpha, sc.fpr_window, cfg.drift.onset)
    # Report admission rate as a METRIC rather than using it to exclude runs.
    # For self-labelling, starvation IS the phenomenon; excluding starved runs
    # truncates the outcome and makes the reported magnitude an artefact of
    # the cut-off.
    post = [r for r in records if r["t"] >= cfg.drift.onset]
    m["admission_rate_post"] = float(np.mean([r["admitted"] for r in post])) if post else float("nan")
    # Realised post-onset prevalence, and the nominal FPR that alarm-rate
    # control predicts at this operating point: f = (alpha - pi*r)/(1 - pi).
    # For arms fed all-sample alarms, fpr_post - fpr_alarm_identity should be
    # ~0; for labelled / reviewed arms, fpr_post should instead be ~alpha.
    pi_hat = float(np.mean([r["label"] for r in post])) if post else float("nan")
    m["prevalence_post"] = pi_hat
    tpr = m.get("tpr_post", float("nan"))
    a = cfg.threshold.alpha
    m["fpr_alarm_identity"] = (float((a - pi_hat * tpr) / (1 - pi_hat))
                               if np.isfinite(tpr) and pi_hat < 1 else float("nan"))
    m["alarm_rate_post"] = float(np.mean([r["flag"] for r in post])) if post else float("nan")
    # Does Proposition 2's ceiling ever bind? It binds only if the drifted
    # nominal (1-alpha) quantile exceeds max B_0. Round-3 review noted this was
    # asserted and never measured; these two numbers measure it per run.
    b0 = np.asarray(cal_scores_initial[-cfg.threshold.window:], dtype=float)
    nom_post = [r["score"] for r in post if r["label"] == 0]
    m["max_B0"] = float(b0.max()) if b0.size else float("nan")
    m["nominal_q_post"] = (float(np.quantile(nom_post, 1 - cfg.threshold.alpha))
                           if nom_post else float("nan"))
    m["ceiling_binds"] = bool(m["nominal_q_post"] > m["max_B0"])
    m["controlled_quantity"] = controlled_quantity(sc.buffer_protocol,
                                                   cfg.threshold.aci_update_on)
    if getattr(sc, "burst_start", -1) >= 0 and sc.burst_len > 0:
        b0, b1 = sc.burst_start, sc.burst_start + sc.burst_len
        inb = [r for r in records if b0 <= r["t"] < b1]
        pre = [r for r in records if cfg.drift.onset <= r["t"] < b0]
        def _tpr(rs):
            a = [r["flag"] for r in rs if r["label"] == 1]
            return float(np.mean(a)) if a else float("nan")
        m["tpr_burst"], m["tpr_pre_burst"] = _tpr(inb), _tpr(pre)
    m["n_adaptations"] = n_adapt
    m["final_h"] = controller.hardness()
    return m, records


def _adapt(head: PatchHead, backbone: FrozenBackbone, buf_imgs: List[torch.Tensor],
           thr: BaseThreshold, controller: BaseController, cfg, device: str,
           rng: np.random.Generator):
    """One adaptation event: re-synthesise from the drifted buffer and
    fine-tune the head for a fixed budget."""
    x = torch.stack(buf_imgs)
    fn = extract(backbone, x, device, cfg.train.batch_size, cfg.train.amp)

    if cfg.stream.couple:
        # THE COUPLING: hardness chosen against the LIVE deployment threshold.
        cal = np.fromiter(thr.buffer, dtype=np.float64)
        per_level = []
        for h in np.linspace(0.0, 1.0, cfg.synth.n_levels):
            xs, _ = synth_mod.synth_batch(x[:32], float(h), cfg.synth, rng)
            fs = extract(backbone, xs, device, cfg.train.batch_size, cfg.train.amp)
            ss, _ = score_features(head, fs, device, cfg.model.topk_frac)
            per_level.append(miss_rate(ss, cal, cfg.curriculum, cfg.threshold.alpha))
        per_level = np.array(per_level)
        k = int(np.argmin(np.abs(per_level - cfg.curriculum.target_miss)))
        controller.update(float(per_level[k]), per_level)
        h_use = float(np.linspace(0.0, 1.0, cfg.synth.n_levels)[k])
    else:
        # DECOUPLED: same trigger, same budget, hardness frozen from training.
        h_use = controller.hardness()

    xs, ms = synth_mod.synth_batch(x, h_use, cfg.synth, rng)
    fs = extract(backbone, xs, device, cfg.train.batch_size, cfg.train.amp)
    ms = F.interpolate(ms, size=(cfg.model.feat_size, cfg.model.feat_size),
                       mode="area")

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.train.lr * 0.3,
                            weight_decay=cfg.train.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    head.train()
    n, bs = fn.shape[0], min(cfg.train.batch_size, fn.shape[0])
    for _ in range(cfg.stream.adapt_steps):
        i = rng.integers(0, n, size=bs)
        j = rng.integers(0, fs.shape[0], size=bs)
        f = torch.cat([fn[i].to(device).float(), fs[j].to(device).float()])
        y = torch.cat([torch.zeros(bs, 1, cfg.model.feat_size, cfg.model.feat_size,
                                   device=device),
                       ms[j].to(device).float()])
        loss = bce(head(f), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head, controller
