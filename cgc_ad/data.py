"""Data loading, leakage-safe splits, and parametric covariate drift.

Design notes
------------
* Images are held in RAM as float32 CHW tensors in [0, 1] at `image_size`.
  A MVTec category is ~350 train + ~120 test images; at 224x224 that is well
  under 200 MB, so RAM caching removes all disk IO from the inner loop.
* The training normals are split THREE ways (train / calib / val). Calibration
  images are never used for gradient updates or model selection. This is the
  leakage control described in F11 of the review.
* Drift is applied on tensors so it can be composed cheaply inside the stream
  simulator.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_mask(path: Optional[str], size: int) -> torch.Tensor:
    if path is None or not os.path.exists(path):
        return torch.zeros(1, size, size)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


@dataclass
class ClassData:
    name: str
    train: torch.Tensor            # N x 3 x S x S  (gradient updates)
    calib: torch.Tensor            # C x 3 x S x S  (conformal thresholds only)
    val: torch.Tensor              # V x 3 x S x S  (model selection only)
    test_images: torch.Tensor      # T x 3 x S x S
    test_labels: torch.Tensor      # T  (0 normal, 1 anomalous)
    test_masks: torch.Tensor       # T x 1 x S x S

    def summary(self) -> Dict[str, int]:
        return {
            "n_train": len(self.train), "n_calib": len(self.calib),
            "n_val": len(self.val), "n_test": len(self.test_images),
            "n_test_anom": int(self.test_labels.sum()),
        }


def load_class(root: str, cls: str, size: int, n_calib: int, n_val: int,
               seed: int = 0, dataset: str = "mvtec") -> ClassData:
    """Load one MVTec-AD (or VisA-in-MVTec-layout) category."""
    cdir = os.path.join(root, cls)
    if not os.path.isdir(cdir):
        raise FileNotFoundError(
            f"{cdir} not found. Expected <root>/<class>/train/good/*.png and "
            f"<root>/<class>/test/<defect>/*.png"
        )

    train_paths = sorted(sum(
        (glob.glob(os.path.join(cdir, "train", "good", f"*.{e}"))
         for e in ("png", "jpg", "JPG", "bmp")), []))
    if not train_paths:
        raise FileNotFoundError(f"no training images under {cdir}/train/good")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_paths))
    n_calib = min(n_calib, max(1, len(idx) // 4))
    n_val = min(n_val, max(1, len(idx) // 8))
    calib_idx = idx[:n_calib]
    val_idx = idx[n_calib:n_calib + n_val]
    train_idx = idx[n_calib + n_val:]

    def stack(ix):
        return torch.stack([_load_image(train_paths[i], size) for i in ix])

    test_imgs, test_lbls, test_masks = [], [], []
    for defect_dir in sorted(glob.glob(os.path.join(cdir, "test", "*"))):
        defect = os.path.basename(defect_dir)
        for p in sorted(sum((glob.glob(os.path.join(defect_dir, f"*.{e}"))
                             for e in ("png", "jpg", "JPG", "bmp")), [])):
            test_imgs.append(_load_image(p, size))
            if defect == "good":
                test_lbls.append(0)
                test_masks.append(torch.zeros(1, size, size))
            else:
                test_lbls.append(1)
                stem = os.path.splitext(os.path.basename(p))[0]
                gt = os.path.join(cdir, "ground_truth", defect, f"{stem}_mask.png")
                if not os.path.exists(gt):
                    gt = os.path.join(cdir, "ground_truth", defect, f"{stem}.png")
                test_masks.append(_load_mask(gt, size))

    return ClassData(
        name=cls,
        train=stack(train_idx), calib=stack(calib_idx), val=stack(val_idx),
        test_images=torch.stack(test_imgs),
        test_labels=torch.tensor(test_lbls, dtype=torch.long),
        test_masks=torch.stack(test_masks),
    )


# --------------------------------------------------------------------------
# Tier-1 parametric covariate drift
# --------------------------------------------------------------------------

_KIND_RANGE = {
    # kind: (value at severity 0, value at severity 1)
    "brightness": (1.0, 1.55),
    "gamma":      (1.0, 0.55),
    "contrast":   (1.0, 0.55),
    "blur":       (0.0, 2.6),     # gaussian sigma in px
    "noise":      (0.0, 0.10),    # additive gaussian std
}


def apply_drift(x: torch.Tensor, kind: str, severity: float,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply drift to a batch (B,3,H,W) or single image (3,H,W), values in [0,1].

    Severity 0 is the identity for every kind, so a stream can ramp smoothly
    from no-drift to full-drift and the pre-onset segment is exactly clean.
    """
    if kind == "none" or severity <= 0:
        return x
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    s = float(np.clip(severity, 0.0, 1.0))

    if kind == "combo":
        for k in ("brightness", "gamma", "blur", "noise"):
            x = apply_drift(x, k, s * 0.75, generator)
        return x.squeeze(0) if single else x

    lo, hi = _KIND_RANGE[kind]
    v = lo + (hi - lo) * s

    if kind == "brightness":
        x = x * v
    elif kind == "gamma":
        x = x.clamp_min(1e-6) ** v
    elif kind == "contrast":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * v + mean
    elif kind == "blur":
        x = _gaussian_blur(x, v)
    elif kind == "noise":
        n = torch.randn(x.shape, generator=generator, device=x.device,
                        dtype=x.dtype)
        x = x + n * v
    else:
        raise ValueError(f"unknown drift kind: {kind}")

    x = x.clamp(0.0, 1.0)
    return x.squeeze(0) if single else x


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-3:
        return x
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return x


def severity_at(step: int, cfg) -> float:
    """Severity schedule for the deployment stream (step change or linear ramp)."""
    if step < cfg.onset:
        return 0.0
    if cfg.ramp <= 0:
        return cfg.severity
    t = (step - cfg.onset) / float(cfg.ramp)
    return float(min(1.0, t) * cfg.severity)
