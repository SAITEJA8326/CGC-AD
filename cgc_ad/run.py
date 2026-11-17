"""Entry point.

    python -m cgc_ad.run --config configs/quick.yaml
    python -m cgc_ad.run --config configs/main.yaml --seeds 0,1,2 --resume \
        --set curriculum.kind=random --tag ctrl_random

Layout produced under `out_dir`:

    results.jsonl                       one row per (run_hash, class, seed)
    runs/<tag>/<hash>/<class>/s<seed>/
        config.json  log.txt  events.jsonl  metrics.csv
        monotonicity.json  result.json  trace.npz
        ckpt/train_latest.pt  ckpt/train_best.pt  ckpt/stream_latest.pt

Everything is append-only or atomically replaced, so a killed Kaggle session
resumes from the last checkpoint instead of starting over.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from .checkpoint import CheckpointManager
from .config import Config, apply_overrides, load_yaml
from .curriculum import build_controller
from .data import load_class
from .engine import (build_banks, evaluate_static, pick_oracle_hardness,
                     run_stream, save_trace, sweep_monotonicity, train_head)
from .model import FrozenBackbone, build_backbone, count_trainable
from .tracking import Tracker, capture_env

_INTERRUPTED = {"flag": False}


def _install_signal_handlers() -> None:
    """Kaggle sends SIGTERM before killing a session. Note it and exit cleanly
    after the current class so the last checkpoint is consistent."""
    def handler(signum, _frame):
        _INTERRUPTED["flag"] = True
        print(f"\n[signal] received {signum}; will stop after the current class.",
              flush=True)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, handler)
        except Exception:
            pass


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def rng_streams(seed: int, cls: str) -> Dict[str, object]:
    """Independent random streams for one (seed, class) run.

    WHY: previously one numpy Generator was threaded through bank synthesis,
    controller construction, head training and stream admission. When the
    feature bank was loaded from cache, synthesis did not consume that
    generator, so training drew different minibatches -- the first arm run in
    a session trained a different head from every later arm. (E1 shows the
    signature: r3_naive at clean AUROC 0.9846, the other five arms at 0.9830.)
    Torch's global RNG, which initialises the head, likewise depended on how
    many classes ran before and on --resume skipping.

    Each consumer now gets its own stream derived from (seed, class), so no
    consumer's draws depend on whether another consumer ran.
    """
    import zlib
    ss = np.random.SeedSequence([int(seed), zlib.crc32(cls.encode()) & 0xFFFFFFFF])
    bank, train, ctrl, stream, oracle, tseed = ss.spawn(6)
    return {"bank": np.random.default_rng(bank),
            "train": np.random.default_rng(train),
            "controller": np.random.default_rng(ctrl),
            "stream": np.random.default_rng(stream),
            "oracle": np.random.default_rng(oracle),
            "torch_seed": int(tseed.generate_state(1)[0])}


def prepare_head(cfg: Config, cd, device: str, tracker=None, rdir: Optional[str] = None):
    """Backbone -> feature bank -> trained (or cached) head.

    Factored out of run_one_class so the determinism guarantee is testable:
    the returned head is a function of (config, class, seed) only.
    """
    from .checkpoint import head_path, load_head, save_head
    from .model import PatchHead
    streams = rng_streams(cfg.seed, cd.name)
    log = tracker.logger if tracker else None

    backbone = build_backbone(cfg, seed=cfg.seed).to(device).eval()
    banks = build_banks(backbone, cd, cfg, streams["bank"], device,
                        tracker=tracker, cls=cd.name)
    controller = build_controller(cfg.curriculum, streams["controller"],
                                  cfg.synth.n_levels)

    hpath = head_path(cfg, cd.name) if cfg.ckpt.head_cache else None
    cached = load_head(hpath) if (hpath and cfg.ckpt.resume) else None
    if cached is not None:
        head = PatchHead(banks.normal.shape[1], cfg.model.head_width).to(device)
        head.load_state_dict(cached["head"])
        controller.load_state_dict(cached["controller"])
        controller.rng.bit_generator.state = cached["controller_rng"]
        tinfo = {**cached["tinfo"], "head_cache": "hit"}
        if log:
            log.info(f"head loaded from cache <- {hpath}")
        return backbone, banks, controller, head, tinfo, streams

    if cfg.curriculum.kind == "oracle":
        controller.set_oracle(pick_oracle_hardness(banks, cfg, device,
                                                   streams["oracle"], tracker))
    train_ckpt = None
    if rdir is not None:
        train_ckpt = CheckpointManager(os.path.join(rdir, "ckpt"),
                                       enabled=cfg.ckpt.enabled,
                                       keep_last=cfg.ckpt.keep_last,
                                       prefix="train", logger=log)
    # Seed torch immediately before the head is initialised, so its initial
    # weights do not depend on backbone construction, oracle probes, class
    # order, or which classes --resume skipped.
    torch.manual_seed(streams["torch_seed"])
    head, tinfo = train_head(banks, cfg, controller, device, streams["train"],
                             tracker=tracker, ckpt=train_ckpt)
    tinfo = {**tinfo, "head_cache": "miss"}
    if hpath:
        save_head(hpath, head, controller, tinfo, log)
    return backbone, banks, controller, head, tinfo, streams


def completed_runs(path: str) -> Set[str]:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        done.add(f"{r['run_hash']}::{r['class']}::{r['seed']}")
                except Exception:
                    continue
    return done


def append_jsonl(path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = json.dumps(row, default=float)
    with open(path, "a") as f:
        f.write(tmp + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_dir_for(cfg: Config, cls: str) -> str:
    return os.path.join(cfg.out_dir, "runs", cfg.tag, cfg.hash(), cls,
                        f"s{cfg.seed}")


def run_one_class(cfg: Config, cls: str, device: str, stage: str) -> Dict:
    t0 = time.time()
    rdir = run_dir_for(cfg, cls)
    resumed = os.path.exists(os.path.join(rdir, "config.json"))
    tr = Tracker(rdir, f"{cfg.tag}/{cls}/s{cfg.seed}", cfg.to_dict(),
                 cfg.logging, resume=resumed)
    try:
        tr.info(f"loading class {cls} from {cfg.data.root}")
        cd = load_class(cfg.data.root, cls, cfg.data.image_size,
                        cfg.data.n_calib, cfg.data.n_val,
                        seed=cfg.seed, dataset=cfg.data.dataset)
        tr.info(f"splits: {cd.summary()}")

        with tr.timer("backbone, banks, head"):
            backbone, banks, controller, head, tinfo, streams = prepare_head(
                cfg, cd, device, tracker=tr, rdir=rdir)
        tr.info(f"backbone: {backbone.describe()}")
        warn = backbone.warn_upsampling()
        if warn:
            tr.warn(warn)

        row: Dict = {
            "run_hash": cfg.hash(), "tag": cfg.tag, "class": cls,
            "seed": cfg.seed, "run_dir": rdir, "config": cfg.to_dict(),
            "env": capture_env(), "data": cd.summary(),
            "trainable_params": count_trainable(head),
            "banks_gb": round(banks.nbytes_gb(), 3), **tinfo,
        }

        # Phase-1 gate: cheap, so run it on every job and never lose the evidence.
        row["monotonicity"] = sweep_monotonicity(head, banks, cfg, device, tr)
        if stage == "gate":
            row["wall_seconds"] = round(time.time() - t0, 1)
            tr.artifact("result", row)
            return row

        # Static evaluation isolates the SCORER.
        with tr.timer("static evaluation"):
            row["static_clean"] = evaluate_static(head, backbone, cd, cfg,
                                                  device, "none", 0.0)
            row["static_drift"] = evaluate_static(head, backbone, cd, cfg, device,
                                                  cfg.drift.kind, cfg.drift.severity)
        row["auroc_retention"] = (row["static_drift"]["image_auroc"]
                                  - row["static_clean"]["image_auroc"])
        tr.scalars(tinfo["steps"], phase="static", **row["static_clean"],
                   drift_auroc=row["static_drift"]["image_auroc"],
                   auroc_retention=row["auroc_retention"])
        if stage == "train":
            row["wall_seconds"] = round(time.time() - t0, 1)
            tr.artifact("result", row)
            return row

        # Stream evaluation isolates the THRESHOLDER and the adaptation loop.
        stream_ckpt = CheckpointManager(os.path.join(rdir, "ckpt"),
                                        enabled=cfg.ckpt.enabled,
                                        keep_last=1, prefix="stream",
                                        logger=tr.logger)
        with tr.timer("deployment stream"):
            metrics, trace = run_stream(head, backbone, cd, banks, controller,
                                        cfg, device, streams["stream"],
                                        tracker=tr, ckpt=stream_ckpt)
        row["stream"] = metrics
        row["controller_history"] = controller.history[-50:]
        row["wall_seconds"] = round(time.time() - t0, 1)
        row.update(tr.gpu_mem())

        if cfg.logging.save_traces:
            save_trace(trace, os.path.join(rdir, "trace.npz"))
        tr.artifact("result", row)
        tr.info(f"done in {row['wall_seconds']}s")
        return row
    finally:
        tr.finish()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("cgc-ad")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated")
    ap.add_argument("--classes", default=None, help="comma-separated override")
    ap.add_argument("--stage", default="full", choices=["gate", "train", "full"],
                    help="gate = monotonicity only; train = + static eval; "
                         "full = + deployment stream")
    ap.add_argument("--resume", action="store_true",
                    help="skip finished (hash, class, seed) triples and resume "
                         "partial ones from their last checkpoint")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and bank caches")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop launching new classes after this many minutes")
    args = ap.parse_args(argv)

    _install_signal_handlers()
    started = time.time()

    def build(seed: int) -> Config:
        c = apply_overrides(load_yaml(args.config), args.overrides)
        if args.tag:
            c.tag = args.tag
        if args.classes:
            c.data.classes = [s for s in args.classes.split(",") if s]
        if args.fresh:
            c.ckpt.resume = False
        c.seed = seed
        return c

    base = build(int(args.seeds.split(",")[0]) if args.seeds else
                 load_yaml(args.config).seed)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [base.seed]

    device = base.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] CUDA unavailable; this will be very slow.", file=sys.stderr)

    os.makedirs(base.out_dir, exist_ok=True)
    os.makedirs(base.ckpt.banks_dir, exist_ok=True)
    results_path = os.path.join(base.out_dir, "results.jsonl")
    done = completed_runs(results_path) if args.resume else set()
    print(f"[info] {len(done)} completed runs found in {results_path}")

    n_ok = n_fail = n_skip = 0
    for seed in seeds:
        cfg = build(seed)
        set_seed(seed, getattr(cfg.train, "deterministic", True))
        h = cfg.hash()
        for cls in cfg.data.classes:
            key = f"{h}::{cls}::{seed}"
            if key in done:
                n_skip += 1
                print(f"[skip] {key}")
                continue
            if _INTERRUPTED["flag"]:
                print("[stop] interrupted; exiting cleanly.")
                return 0
            if args.time_budget_min and \
                    (time.time() - started) / 60 > args.time_budget_min:
                print(f"[stop] time budget of {args.time_budget_min} min reached; "
                      f"rerun with --resume to continue.")
                return 0

            print(f"[run ] hash={h} seed={seed} class={cls} stage={args.stage} "
                  f"ctrl={cfg.curriculum.kind} thr={cfg.threshold.policy} "
                  f"couple={cfg.stream.couple}")
            try:
                row = run_one_class(cfg, cls, device, args.stage)
            except Exception as e:
                import traceback
                traceback.print_exc()
                append_jsonl(results_path,
                             {"run_hash": h, "tag": cfg.tag, "class": cls,
                              "seed": seed, "stage": args.stage,
                              "error": repr(e)})
                n_fail += 1
                continue
            row["stage"] = args.stage
            append_jsonl(results_path, row)
            n_ok += 1
            if "stream" in row:
                m = row["stream"]
                print(f"       AUROC {row['static_clean']['image_auroc']:.4f} "
                      f"-> {row['static_drift']['image_auroc']:.4f} | "
                      f"MAD-FPR {m['mad_fpr']:.4f} | TPR_post {m['tpr_post']:.3f} "
                      f"| adapts {m['n_adaptations']} | {row['wall_seconds']}s")
            else:
                mo = row["monotonicity"]
                print(f"       gate spearman={mo['spearman']:.3f} "
                      f"range={mo['range']:.3f} pass={mo['gate_pass']}")

    print(f"[done] ok={n_ok} failed={n_fail} skipped={n_skip} -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
