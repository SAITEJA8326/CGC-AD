"""Experiment tracking and logging.

One `Tracker` per (config, class, seed). It owns a run directory and writes
four things, all append-only so a killed Kaggle session loses nothing:

    <run_dir>/config.json     the exact config, plus environment capture
    <run_dir>/log.txt         human-readable log (also mirrored to stdout)
    <run_dir>/events.jsonl    one JSON object per logged step
    <run_dir>/metrics.csv     flat table for pandas / plotting

TensorBoard and Weights & Biases are optional and degrade to no-ops if the
package is missing, which is the normal state on a bare Kaggle image.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None


def capture_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            env["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        pass
    return env


class Tracker:
    def __init__(self, run_dir: str, run_name: str, config: Dict[str, Any],
                 log_cfg=None, resume: bool = False):
        self.run_dir = run_dir
        self.run_name = run_name
        self.cfg = log_cfg
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

        self.events_path = os.path.join(run_dir, "events.jsonl")
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self._csv_keys: Optional[list] = None

        # ---- python logging: file + stdout ------------------------------
        self.logger = logging.getLogger(f"cgc_ad.{run_name}")
        self.logger.handlers.clear()
        self.logger.propagate = False
        level = getattr(logging, str(getattr(log_cfg, "level", "INFO")).upper(),
                        logging.INFO)
        self.logger.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                "%H:%M:%S")
        fh = logging.FileHandler(os.path.join(run_dir, "log.txt"), mode="a")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        # ---- config + environment ---------------------------------------
        if not resume or not os.path.exists(os.path.join(run_dir, "config.json")):
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump({"run_name": run_name, "config": config,
                           "env": capture_env()}, f, indent=2, default=str)

        self._t0 = time.time()
        self._tb = self._init_tb()
        self._wandb = self._init_wandb(config)
        self.info(f"run_dir={run_dir}" + (" (resumed)" if resume else ""))

    # ---- optional backends ----------------------------------------------
    def _init_tb(self):
        if not getattr(self.cfg, "tensorboard", False):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(os.path.join(self.run_dir, "tb"))
        except Exception as e:
            self.warn(f"tensorboard unavailable ({e!r}); continuing without it")
            return None

    def _init_wandb(self, config):
        if not getattr(self.cfg, "wandb", False):
            return None
        try:
            import wandb
            return wandb.init(project=getattr(self.cfg, "wandb_project", "cgc-ad"),
                              name=self.run_name, config=config,
                              reinit=True, resume="allow")
        except Exception as e:
            self.warn(f"wandb unavailable ({e!r}); continuing without it")
            return None

    # ---- logging ---------------------------------------------------------
    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warn(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def scalars(self, step: int, phase: str = "train", **kv: Any) -> None:
        """Record numeric metrics for one step. Goes to every active sink."""
        rec = {"step": int(step), "phase": phase,
               "wall": round(time.time() - self._t0, 2), **kv}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        self._csv_append(rec)
        if self._tb is not None:
            for k, v in kv.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"{phase}/{k}", v, step)
        if self._wandb is not None:
            self._wandb.log({f"{phase}/{k}": v for k, v in kv.items()}, step=step)

    def _csv_append(self, rec: Dict[str, Any]) -> None:
        new = not os.path.exists(self.csv_path)
        if self._csv_keys is None:
            self._csv_keys = list(rec.keys()) if new else \
                next(csv.reader(open(self.csv_path)))
        row = {k: rec.get(k, "") for k in self._csv_keys}
        for k in rec:                                  # tolerate late-arriving keys
            if k not in self._csv_keys:
                row[f"extra_{k}"] = rec[k]
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def artifact(self, name: str, obj: Any) -> str:
        """Dump a small JSON-serialisable artifact next to the run."""
        path = os.path.join(self.run_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=float)
        return path

    @contextmanager
    def timer(self, name: str):
        t = time.time()
        yield
        self.info(f"{name} took {time.time() - t:.1f}s")

    def gpu_mem(self) -> Dict[str, float]:
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            return {"gpu_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                    "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
        except Exception:
            return {}

    def finish(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
        for h in self.logger.handlers:
            h.close()
        self.logger.handlers.clear()
