"""Shared fixtures.

Tests are split into two tiers so the fast tier runs anywhere:

  * NO-GPU / NO-TORCH tier -- conformal maths, thresholds, controllers,
    metrics, config. Pure numpy. Runs in seconds on a laptop or in CI.
  * TORCH tier -- synthesis, backbone, checkpoint round-trips. Skipped
    automatically when torch is unavailable.

Run the fast tier before every commit; run the full suite once per session on
Kaggle, where torch is present.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip if False else None
try:
    import torch as _torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def cfg():
    from cgc_ad.config import Config
    return Config()


@pytest.fixture
def shifted_stream(rng):
    """N(0,1) for 800 steps, then N(1,1). The canonical drift test case."""
    pre = rng.normal(0.0, 1.0, size=800)
    post = rng.normal(1.0, 1.0, size=3200)
    return np.concatenate([pre, post]), 800
