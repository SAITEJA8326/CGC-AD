"""Difficulty-parameterised synthetic anomalies (CutPaste / NSA family).

A single scalar `h in [0, 1]` controls every salience axis simultaneously:

    axis            h = 0 (easy)          h = 1 (hard)
    -----------     ------------------    ---------------------------
    patch area      large                 small
    blend opacity   opaque                translucent
    colour delta    strong                subtle
    edge feather    hard edge             soft, blended edge
    source patch    distant / dissimilar  nearby / self-similar

IMPORTANT (review F9): this construction guarantees monotone *salience*, not
monotone *detectability*. `sweep_monotonicity` in engine.py is the Phase-1
gate that checks the latter empirically. Do not build the controller on top
of an unverified hardness axis.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def hardness_ladder(n_levels: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_levels)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def synthesize(img: torch.Tensor, h: float, cfg,
               rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (anomalous image, binary mask) for one normal image (3,H,W)."""
    h = float(np.clip(h, 0.0, 1.0))
    C, H, W = img.shape
    out = img.clone()
    mask = torch.zeros(1, H, W, dtype=img.dtype)

    area_frac = _lerp(cfg.area_easy, cfg.area_hard, h)
    alpha = _lerp(cfg.alpha_easy, cfg.alpha_hard, h)
    jitter = _lerp(cfg.jitter_easy, cfg.jitter_hard, h)
    feather = int(round(_lerp(0.0, cfg.feather_hard, h)))
    # Harder => source patch is taken from nearby (self-similar) content.
    max_shift = int(_lerp(0.9, 0.12, h) * min(H, W))

    n_patches = int(rng.integers(1, cfg.max_patches + 1))
    for _ in range(n_patches):
        aspect = float(np.exp(rng.uniform(-0.7, 0.7)))
        area = area_frac * H * W / n_patches
        ph = int(np.clip(np.sqrt(area * aspect), 4, H - 2))
        pw = int(np.clip(np.sqrt(area / aspect), 4, W - 2))

        # destination
        dy = int(rng.integers(0, H - ph))
        dx = int(rng.integers(0, W - pw))
        # source, constrained to be within max_shift of the destination
        sy = int(np.clip(dy + rng.integers(-max_shift, max_shift + 1), 0, H - ph))
        sx = int(np.clip(dx + rng.integers(-max_shift, max_shift + 1), 0, W - pw))

        patch = img[:, sy:sy + ph, sx:sx + pw].clone()
        if rng.random() < 0.5:
            patch = torch.flip(patch, dims=[2])
        if rng.random() < 0.3:
            patch = torch.rot90(patch, k=int(rng.integers(1, 4)), dims=[1, 2])
            patch = F.interpolate(patch.unsqueeze(0), size=(ph, pw),
                                  mode="bilinear", align_corners=False).squeeze(0)

        # photometric perturbation
        delta = torch.from_numpy(
            rng.uniform(-jitter, jitter, size=(C, 1, 1)).astype(np.float32))
        patch = (patch + delta).clamp(0.0, 1.0)

        m = torch.ones(1, ph, pw, dtype=img.dtype)
        if feather > 0:
            m = _feather(m, feather)

        region = out[:, dy:dy + ph, dx:dx + pw]
        blend = alpha * m
        out[:, dy:dy + ph, dx:dx + pw] = region * (1 - blend) + patch * blend
        mask[:, dy:dy + ph, dx:dx + pw] = torch.maximum(
            mask[:, dy:dy + ph, dx:dx + pw], (m > 0.5).to(img.dtype))

    return out.clamp(0.0, 1.0), mask


def _feather(m: torch.Tensor, r: int) -> torch.Tensor:
    """Soft-edge a binary patch mask with a box blur of radius r."""
    k = 2 * r + 1
    m = F.avg_pool2d(F.pad(m.unsqueeze(0), (r, r, r, r), mode="constant", value=0.0),
                     kernel_size=k, stride=1).squeeze(0)
    return m.clamp(0.0, 1.0)


def synth_batch(images: torch.Tensor, h: float, cfg,
                rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ms = [], []
    for i in range(images.shape[0]):
        x, m = synthesize(images[i], h, cfg, rng)
        xs.append(x)
        ms.append(m)
    return torch.stack(xs), torch.stack(ms)
