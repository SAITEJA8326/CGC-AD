"""Pipeline tests: splits, synthesis, metrics, determinism, config hashing.

Torch-dependent tests skip cleanly when torch is absent, so the numpy-only
suite still runs anywhere. On Kaggle everything runs.
"""
from __future__ import annotations

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from cgc_ad.config import Config, apply_overrides, from_dict

torch = pytest.importorskip("torch", reason="torch-dependent pipeline tests")


# ==========================================================================
# 1. Config identity  (protects --resume)
# ==========================================================================

def test_hash_ignores_bookkeeping_fields():
    """Changing out_dir / logging / device must NOT invalidate finished work.

    WHY: --resume keys on the run hash. If the hash moved when you changed a
    log directory between Kaggle sessions, every completed run would silently
    be recomputed and results.jsonl would fill with duplicates under different
    hashes -- which then double-counts in the paired statistics.
    """
    a = Config()
    b = Config()
    b.out_dir = "/somewhere/else"
    b.logging.tensorboard = True
    b.ckpt.every_steps = 999
    b.device = "cpu"
    assert a.hash() == b.hash()


def test_hash_changes_for_every_scientific_field():
    """Any field that can change the result must change the hash.

    WHY: the converse failure is worse -- two different experiments sharing a
    hash means --resume skips the second one and you unknowingly analyse the
    first one twice.
    """
    base = Config().hash()
    for path, val in [("curriculum.kind", "random"),
                      ("threshold.policy", "window"),
                      ("threshold.alpha", 0.1),
                      ("stream.couple", False),
                      ("stream.contamination", 0.05),
                      ("stream.admit_explore", 0.05),
                      ("drift.severity", 0.3),
                      ("train.max_steps", 1234),
                      ("model.backbone", "wide_resnet50_2"),
                      ("synth.n_levels", 9),
                      ("seed", 7)]:
        c = apply_overrides(Config(), [f"{path}={val}"])
        assert c.hash() != base, f"{path} does not affect the run hash"


def test_config_round_trips_through_json():
    """results.jsonl stores the config; it must reload exactly.

    WHY: the audit tool reconstructs what each sweep actually varied from these
    stored configs. That is how the invalid contamination sweep was caught.
    """
    c = apply_overrides(Config(), ["curriculum.kind=random", "threshold.alpha=0.1"])
    assert from_dict(json.loads(json.dumps(c.to_dict()))).hash() == c.hash()


# ==========================================================================
# 2. Leakage-safe splits
# ==========================================================================

def _fake_mvtec(root, cls="widget", n_train=120, n_test_good=20, n_test_bad=20,
                size=64):
    from PIL import Image
    rng = np.random.default_rng(0)
    def w(path, arr):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(arr.astype(np.uint8)).save(path)
    for i in range(n_train):
        w(f"{root}/{cls}/train/good/{i:03d}.png",
          rng.integers(90, 140, (size, size, 3)))
    for i in range(n_test_good):
        w(f"{root}/{cls}/test/good/{i:03d}.png",
          rng.integers(90, 140, (size, size, 3)))
    for i in range(n_test_bad):
        a = rng.integers(90, 140, (size, size, 3)); a[20:40, 20:40] = 255
        w(f"{root}/{cls}/test/defect/{i:03d}.png", a)
        m = np.zeros((size, size)); m[20:40, 20:40] = 255
        w(f"{root}/{cls}/ground_truth/defect/{i:03d}_mask.png", m)
    return cls


def test_train_calib_val_splits_are_disjoint():
    """The three normal splits must not share a single image.

    WHY: the calibration split sets tau and the validation split selects the
    model. Any overlap with train leaks the training set into the threshold,
    which inflates every FPR-tracking number in the paper and is the single
    hardest error to detect after the fact.
    """
    from cgc_ad.data import load_class
    with tempfile.TemporaryDirectory() as d:
        cls = _fake_mvtec(d)
        cd = load_class(d, cls, 64, n_calib=30, n_val=15, seed=0)
        def keys(t):
            return {hash(x.numpy().tobytes()) for x in t}
        tr, ca, va = keys(cd.train), keys(cd.calib), keys(cd.val)
        assert not (tr & ca) and not (tr & va) and not (ca & va)
        assert len(tr) + len(ca) + len(va) == 120


def test_splits_are_deterministic_given_the_seed():
    """Same seed => same split, or nothing is reproducible."""
    from cgc_ad.data import load_class
    with tempfile.TemporaryDirectory() as d:
        cls = _fake_mvtec(d)
        a = load_class(d, cls, 64, 30, 15, seed=3)
        b = load_class(d, cls, 64, 30, 15, seed=3)
        c = load_class(d, cls, 64, 30, 15, seed=4)
        assert torch.equal(a.calib, b.calib)
        assert not torch.equal(a.calib, c.calib)


def test_calibration_size_is_capped_by_available_data():
    """Requesting more calibration images than exist must not silently starve
    the training split.

    WHY: `toothbrush` has 60 normals. Asking for n_calib=60 would leave zero
    training images. The loader caps at len//4, which is why toothbrush ended
    up with n_calib=15 and an unattainable alpha=0.05 -- a real event in the
    results that this test makes visible up front.
    """
    from cgc_ad.data import load_class
    with tempfile.TemporaryDirectory() as d:
        cls = _fake_mvtec(d, n_train=60)
        cd = load_class(d, cls, 64, n_calib=200, n_val=100, seed=0)
        assert len(cd.calib) <= 15 and len(cd.train) > 0
        alpha_min = 1.0 / (len(cd.calib) + 1)
        assert alpha_min > 0.05, "this class cannot support alpha=0.05"


# ==========================================================================
# 3. Drift and synthesis
# ==========================================================================

def test_zero_severity_is_the_identity():
    """The pre-onset segment of every stream must be exactly undrifted.

    WHY: if severity 0 were not the identity, the "pre-drift" baseline FPR
    would already be contaminated and the recovery-delay metric would be
    measured from the wrong origin.
    """
    from cgc_ad.data import apply_drift
    x = torch.rand(4, 3, 32, 32)
    for k in ("brightness", "gamma", "blur", "noise", "contrast", "combo"):
        assert torch.equal(apply_drift(x, k, 0.0), x)


def test_drift_is_monotone_in_severity():
    """Higher severity must move the image further from the original.

    WHY: severity is the x-axis of the benefit-vs-drift figure. A non-monotone
    knob makes that figure uninterpretable.
    """
    from cgc_ad.data import apply_drift
    x = torch.rand(2, 3, 32, 32) * 0.6 + 0.2
    for k in ("brightness", "gamma", "blur", "noise"):
        d = [float((apply_drift(x, k, s) - x).abs().mean())
             for s in (0.2, 0.5, 0.9)]
        assert d[0] < d[1] < d[2], k


def test_synth_mask_matches_the_modified_pixels():
    """The mask must cover where the image actually changed.

    WHY: the mask is the training target. A misaligned mask trains the head to
    fire in the wrong place, and pixel AUROC would still look plausible.
    """
    from cgc_ad.config import SynthConfig
    from cgc_ad.synth import synthesize
    rng = np.random.default_rng(0)
    img = torch.rand(3, 64, 64)
    x, m = synthesize(img, 0.0, SynthConfig(), rng)
    changed = (x - img).abs().sum(0) > 1e-4
    assert m.shape == (1, 64, 64)
    assert 0 < m.sum() < m.numel()
    covered = (changed & (m[0] > 0.5)).sum() / max(1, changed.sum())
    assert covered > 0.8


def test_synth_salience_decreases_with_hardness():
    """Higher h must produce a smaller pixel perturbation.

    WHY: this is the PRECONDITION for the monotonicity gate. If salience is not
    monotone in h, no amount of model training will make detectability monotone,
    and the setpoint controller is chasing an ill-posed target. In the real runs
    5 of 15 classes failed the downstream gate; this test confirms the
    generator is not the cause.
    """
    from cgc_ad.config import SynthConfig
    from cgc_ad.synth import synthesize
    cfg = SynthConfig()
    img = torch.rand(3, 96, 96)
    e = []
    for h in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = [float((synthesize(img, h, cfg, np.random.default_rng(s))[0] - img)
                   .abs().sum()) for s in range(12)]
        e.append(float(np.mean(d)))
    assert all(a > b for a, b in zip(e, e[1:])), e


def test_synth_is_deterministic_given_the_rng():
    from cgc_ad.config import SynthConfig
    from cgc_ad.synth import synthesize
    img = torch.rand(3, 64, 64)
    a, _ = synthesize(img, 0.4, SynthConfig(), np.random.default_rng(7))
    b, _ = synthesize(img, 0.4, SynthConfig(), np.random.default_rng(7))
    assert torch.equal(a, b)


# ==========================================================================
# 4. Stream metrics
# ==========================================================================

def _records(n=8000, onset=4000, fpr_pre=0.05, fpr_post=0.30, tpr=0.8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        label = int(rng.random() < 0.05)
        p = tpr if label else (fpr_pre if t < onset else fpr_post)
        out.append({"t": t, "score": float(rng.normal()),
                    "flag": int(rng.random() < p), "label": label,
                    "tau": 1.0, "alpha_t": 0.05, "severity": 0.0,
                    "admitted": 1, "martingale": 1.0, "h": 0.3})
    return out


def test_stream_metrics_recover_the_planted_rates():
    """The metric code must report back what was planted.

    WHY: MAD-FPR is the pre-registered primary endpoint. A bug in the windowing
    or in the pre/post split would move the headline number without changing
    anything else, and nothing else in the pipeline would notice.
    """
    from cgc_ad.engine import stream_metrics
    rec = _records()
    m = stream_metrics(rec, alpha=0.05, window=100, onset=4000)

    # Compare against the rates actually realised in the fixture, not against
    # the nominal ones. At 5% prevalence a 1000-step fixture holds only ~22
    # post-onset anomalies, so a nominal-rate assertion has a binomial SE of
    # 0.085 and fails on RNG rather than on a metric bug. This asserts the
    # metric reproduces the fixture exactly, then checks the fixture itself is
    # close to nominal now that it is large enough to be.
    lbl = np.array([r["label"] for r in rec])
    flg = np.array([r["flag"] for r in rec], dtype=float)
    post = np.arange(len(rec)) >= 4000
    emp_fpr = flg[(lbl == 0) & post].mean()
    emp_tpr = flg[(lbl == 1) & post].mean()
    assert m["fpr_post"] == pytest.approx(emp_fpr)
    assert m["tpr_post"] == pytest.approx(emp_tpr)
    assert abs(emp_fpr - 0.30) < 0.03
    assert abs(emp_tpr - 0.80) < 0.06
    assert m["mad_fpr"] > 0.15


def test_mad_fpr_is_near_zero_for_a_well_calibrated_stream():
    from cgc_ad.engine import stream_metrics
    m = stream_metrics(_records(fpr_pre=0.05, fpr_post=0.05),
                       alpha=0.05, window=100, onset=500)
    assert m["mad_fpr"] < 0.04


def test_auroc_is_blind_to_the_threshold_policy():
    """AUROC must NOT respond to a pure threshold change.

    WHY: this is review finding F3 as an executable check. AUROC was the
    original headline metric, and it is mathematically incapable of measuring
    the thresholding component. Encoding that here stops anyone from quietly
    reintroducing it as the primary endpoint.
    """
    from cgc_ad.engine import stream_metrics
    base = _records()
    shifted = [dict(r, flag=1 - r["flag"]) for r in base]      # flip every alarm
    a = stream_metrics(base, 0.05, 100, 500)
    b = stream_metrics(shifted, 0.05, 100, 500)
    assert abs(a["stream_auroc"] - b["stream_auroc"]) < 1e-12
    assert abs(a["fpr_post"] - b["fpr_post"]) > 0.3


# ==========================================================================
# 5. Checkpoint / resume determinism
# ==========================================================================

def test_rng_state_round_trips():
    """Resuming must reproduce the identical random stream.

    WHY: without this, a run interrupted by a Kaggle session kill produces
    different numbers from an uninterrupted one, and results.jsonl silently
    mixes two experiments.
    """
    from cgc_ad.checkpoint import rng_state, set_rng_state
    rng = np.random.default_rng(0)
    _ = rng.normal(size=100)
    st = rng_state(rng)
    a = rng.normal(size=50)
    set_rng_state(st, rng)
    assert np.allclose(a, rng.normal(size=50))


def test_atomic_save_leaves_no_partial_file():
    from cgc_ad.checkpoint import atomic_save, safe_load
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pt")
        atomic_save({"x": torch.arange(5)}, p)
        assert not os.path.exists(p + ".tmp")
        assert torch.equal(safe_load(p)["x"], torch.arange(5))


def test_safe_load_survives_a_truncated_checkpoint():
    """A checkpoint truncated by a hard kill must not crash the resume path."""
    from cgc_ad.checkpoint import safe_load
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "broken.pt")
        open(p, "wb").write(b"\x80\x02}q")
        assert safe_load(p) is None
        assert safe_load(os.path.join(d, "absent.pt")) is None
