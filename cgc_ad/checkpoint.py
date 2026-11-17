"""Checkpointing and the feature-bank disk cache.

Kaggle sessions are killed at the 9–12 h wall and lose the container. Three
things therefore have to survive to disk:

1. **Feature banks** — the only expensive GPU work in the pipeline. Cached by a
   hash over every config field that can change the features. Rebuilt only when
   that hash changes.
2. **Training state** — head weights, optimiser, AMP scaler, controller, step
   counter, and all RNG streams. Restoring these makes a resumed run
   bit-identical to an uninterrupted one, which is what "reproducible" has to
   mean if any run ever gets interrupted.
3. **Stream state** — head, threshold policy, martingale, buffer, and the
   partial record list, so a 1500-step deployment simulation can be resumed
   mid-flight.

Every write goes through `atomic_save`: write to `.tmp`, fsync, rename. A
session killed during a save leaves the previous checkpoint intact.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------
# Atomic IO
# --------------------------------------------------------------------------

def atomic_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_load(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        # A checkpoint truncated by a hard kill: fall back rather than crash.
        return None


# --------------------------------------------------------------------------
# RNG state
# --------------------------------------------------------------------------

def rng_state(np_rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    st: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python_np_legacy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if np_rng is not None:
        st["np_gen"] = np_rng.bit_generator.state
    return st


def set_rng_state(st: Dict[str, Any],
                  np_rng: Optional[np.random.Generator] = None) -> None:
    if st is None:
        return
    if "torch" in st:
        torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:
            pass
    if "python_np_legacy" in st:
        np.random.set_state(st["python_np_legacy"])
    if np_rng is not None and "np_gen" in st:
        np_rng.bit_generator.state = st["np_gen"]


# --------------------------------------------------------------------------
# Checkpoint manager
# --------------------------------------------------------------------------

class CheckpointManager:
    """Rotating step checkpoints plus an optional `best.pt`."""

    def __init__(self, ckpt_dir: str, enabled: bool = True, keep_last: int = 2,
                 prefix: str = "train", logger=None):
        self.dir = ckpt_dir
        self.enabled = enabled
        self.keep_last = max(1, keep_last)
        self.prefix = prefix
        self.log = logger
        os.makedirs(ckpt_dir, exist_ok=True)

    def _p(self, name: str) -> str:
        return os.path.join(self.dir, f"{self.prefix}_{name}.pt")

    def save(self, state: Dict[str, Any], step: int) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._p(f"step{step:07d}")
        atomic_save({"step": step, **state}, path)
        atomic_save({"step": step, **state}, self._p("latest"))
        self._rotate()
        if self.log:
            self.log.info(f"checkpoint saved: {os.path.basename(path)}")
        return path

    def save_best(self, state: Dict[str, Any], metric: float, step: int,
                  higher_better: bool = True) -> bool:
        if not self.enabled:
            return False
        cur = self.read_best_metric()
        better = (cur is None or
                  (metric > cur if higher_better else metric < cur))
        if better:
            atomic_save({"step": step, "metric": float(metric), **state},
                        self._p("best"))
            with open(self._p("best") + ".meta.json", "w") as f:
                json.dump({"metric": float(metric), "step": step}, f)
            if self.log:
                self.log.info(f"new best ({metric:.5f}) at step {step}")
        return better

    def read_best_metric(self) -> Optional[float]:
        m = self._p("best") + ".meta.json"
        if not os.path.exists(m):
            return None
        try:
            return float(json.load(open(m))["metric"])
        except Exception:
            return None

    def load_latest(self) -> Optional[Dict[str, Any]]:
        return safe_load(self._p("latest"))

    def load_best(self) -> Optional[Dict[str, Any]]:
        return safe_load(self._p("best"))

    def _rotate(self) -> None:
        steps = sorted(f for f in os.listdir(self.dir)
                       if f.startswith(f"{self.prefix}_step") and f.endswith(".pt"))
        for f in steps[:-self.keep_last]:
            try:
                os.remove(os.path.join(self.dir, f))
            except OSError:
                pass

    def clear(self) -> None:
        for f in os.listdir(self.dir):
            if f.startswith(self.prefix + "_"):
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass


# --------------------------------------------------------------------------
# Feature-bank disk cache
# --------------------------------------------------------------------------

def banks_key(cfg, cls: str) -> str:
    """Hash over every field that can change the extracted features.

    Deliberately excludes the training, curriculum, threshold, stream, and
    drift sections: those never touch feature extraction, so one cache serves
    every arm of the sweep. That is what makes the ablation grid cheap.
    """
    payload = {
        "class": cls,
        "data": {k: getattr(cfg.data, k) for k in
                 ("root", "dataset", "image_size", "n_calib", "n_val")},
        "model": {k: getattr(cfg.model, k) for k in
                  ("backbone", "layers", "stages", "feat_size",
                   "proj_dim", "pretrained")},
        "synth": cfg.synth.__dict__,
        "seed": cfg.seed,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def banks_path(cfg, cls: str) -> str:
    return os.path.join(cfg.ckpt.banks_dir, f"banks_{cls}_{banks_key(cfg, cls)}.pt")


def save_banks(banks, path: str, logger=None) -> None:
    from dataclasses import asdict
    atomic_save({"levels": banks.levels,
                 "normal": banks.normal, "calib": banks.calib,
                 "val_normal": banks.val_normal,
                 "synth": banks.synth, "synth_masks": banks.synth_masks,
                 "val_synth": banks.val_synth}, path)
    if logger:
        logger.info(f"feature bank cached -> {path} "
                    f"({os.path.getsize(path) / 1e9:.2f} GB)")


def load_banks(path: str, logger=None):
    d = safe_load(path)
    if d is None:
        return None
    from .engine import Banks
    if logger:
        logger.info(f"feature bank loaded <- {path} "
                    f"({os.path.getsize(path) / 1e9:.2f} GB)")
    return Banks(normal=d["normal"], synth=d["synth"],
                 synth_masks=d["synth_masks"], levels=d["levels"],
                 calib=d["calib"], val_normal=d["val_normal"],
                 val_synth=d["val_synth"])


def disk_free_gb(path: str) -> float:
    try:
        os.makedirs(path, exist_ok=True)
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float("inf")
