"""Findings from the controlled simulation, pinned.

WHY THIS FILE EXISTS

Round 4 found two things the MVTec runs could not show:

  * EROSION. Under score-thresholded admission with FIFO eviction, the buffer
    maximum strictly decreases every time its current maximum is evicted,
    because every replacement was admitted below it. So self-labelling plus the
    corrected feedback (P1) is still a ratchet -- a slow one, whose damage grows
    with stream length even with no drift. This reverses the 6-page draft's
    claim that "P1 alone suffices while drift stays below the ceiling".
  * MASKING. MVTec streams resample 20-60 test images per class. Repeated
    scores let the maximum be re-admitted, so erosion almost vanishes there.
    That is why P1-only looked fine on MVTec (FPR 0.0196 in R2).

These tests use `cgc_ad.simulate`, which drives the real threshold code.
"""
import numpy as np
import pytest

from cgc_ad.simulate import simulate_arm


def _fpr(arm, regime="stationary", seeds=4, **kw):
    return float(np.mean([simulate_arm(arm, regime, s, **kw)["fpr"]
                          for s in range(seeds)]))


def test_p1_only_erodes_with_stream_length_and_no_drift():
    """Damage grows with horizon: the ceiling max(B_t) keeps falling."""
    short = _fpr("p1_only", T=1500, onset=500)
    long = _fpr("p1_only", T=12000, onset=6000)
    assert short < 0.12
    assert long > 0.2 and long > short + 0.1, (short, long)


def test_exploration_stops_the_erosion():
    """P2 is what replenishes the upper tail. With it, the long-horizon FPR
    stays bounded -- so exploration IS necessary, on long streams, even without
    drift (the opposite of what the 6-page draft concluded)."""
    assert _fpr("patch", T=12000, onset=6000) < 0.06


def test_resampling_a_small_image_pool_masks_the_erosion():
    """Why MVTec could not reveal it: 20 recurring nominal images."""
    fresh = _fpr("p1_only", T=12000, onset=6000)
    pooled = _fpr("p1_only", T=12000, onset=6000, pool=20)
    assert pooled < fresh - 0.1, (fresh, pooled)


def test_ceiling_binds_by_design_and_blocks_p1_only():
    r = [simulate_arm("p1_only", "drift", s) for s in range(4)]
    assert all(x["ceiling_binds"] for x in r)
    assert np.mean([x["fpr"] for x in r]) > 0.8


def test_static_threshold_fails_under_drift_in_the_controlled_setting():
    """The motivating premise, which the R3 MVTec runs never established
    (their drift moved clean AUROC by -0.001)."""
    assert _fpr("static", "drift") > 0.5


@pytest.mark.parametrize("regime", ["stationary", "drift", "burst"])
def test_reviewed_and_labelled_hold_alpha_in_every_regime(regime):
    for arm in ("labelled", "reviewed"):
        assert _fpr(arm, regime) == pytest.approx(0.05, abs=0.012), (arm, regime)


def test_alarm_rate_arms_desensitise_in_bursts_and_reviewed_does_not():
    def tpr(arm, regime):
        return float(np.mean([simulate_arm(arm, regime, s)["tpr"] for s in range(4)]))
    for arm in ("alarmrate", "patch"):
        assert tpr(arm, "burst") < tpr(arm, "stationary") - 0.15, arm
    assert tpr("reviewed", "burst") > tpr("reviewed", "stationary") - 0.05


def test_every_arm_sees_the_same_stream():
    """Pairing: arms differ only in the threshold layer."""
    a = simulate_arm("labelled", "drift", 3)
    b = simulate_arm("naive", "drift", 3)
    assert a["max_B0"] == b["max_B0"]
    assert a["shift_trace"] == b["shift_trace"]
