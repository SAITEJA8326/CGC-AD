"""Online threshold policies, including the two structural failure modes.

WHY THESE TESTS EXIST
Both defects found in the first result set were invisible to any accuracy
metric: the detector reported a clean-looking FPR log while being either
switched off (tau = inf) or permanently saturated (the ratchet). Each is now
pinned by a test that fails loudly if the behaviour returns.
"""
import numpy as np
import pytest
from cgc_ad.config import Config
from cgc_ad.conformal import build_threshold

POLICIES = ["static", "window", "aci", "robust_aci"]


def _run(cfg, scores, protocol="clean", explore=0.0, seed=0):
    rng = np.random.default_rng(seed)
    thr = build_threshold(cfg.threshold, np.random.default_rng(7).normal(size=200))
    flags, taus, admits = [], [], []
    for s in scores:
        f = thr.flag(s)
        admit = True if protocol == "clean" else ((not f) or rng.random() < explore)
        thr.observe(s, admit, flagged=f)
        flags.append(f); taus.append(thr.tau); admits.append(admit)
    return np.array(flags), np.array(taus), np.array(admits), thr


@pytest.mark.parametrize("policy", POLICIES)
def test_no_drift_gives_nominal_fpr(policy):
    """Sanity floor: with a stationary stream every policy must sit at alpha.
    A policy that cannot do this has no business being compared under drift."""
    cfg = Config(); cfg.threshold.policy = policy
    rng = np.random.default_rng(0)
    flags, _, _, _ = _run(cfg, rng.normal(size=3000))
    assert 0.02 <= flags[-1500:].mean() <= 0.10, (policy, flags[-1500:].mean())


@pytest.mark.parametrize("policy", ["window", "aci", "robust_aci"])
def test_adaptive_policies_recover_after_a_shift(policy, shifted_stream):
    """The core functional claim of component (b). Static is expected to fail
    this and is excluded on purpose -- that expectation is asserted below."""
    cfg = Config(); cfg.threshold.policy = policy; cfg.threshold.gamma_aci = 0.01
    scores, onset = shifted_stream
    flags, _, _, _ = _run(cfg, scores)
    assert flags[-1000:].mean() <= 0.15, (policy, flags[-1000:].mean())


def test_static_threshold_fails_under_shift():
    """A negative control. If static ever passed, the drift protocol would be
    too weak to distinguish anything and every comparison would be vacuous."""
    cfg = Config(); cfg.threshold.policy = "static"
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(size=800), rng.normal(1.0, 1.0, size=3200)]
    flags, _, _, _ = _run(cfg, scores)
    assert flags[-1000:].mean() > 0.20


def test_aci_error_indicator_is_degenerate_when_coupled_to_admission():
    """Pins the DEGENERACY as a documented behaviour, not a lurking bug.

    Under self-labelling, admitted := not flagged, so 1[s > tau] is identically
    zero on admitted samples and alpha_t ramps monotonically to its cap. This
    test asserts the broken behaviour still occurs in the `admitted` ablation
    arm, so the paper's claim about it stays true if the code changes.
    """
    cfg = Config()
    cfg.threshold.policy = "aci"
    cfg.threshold.aci_update_on = "admitted"
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(size=800), rng.normal(1.0, 1.0, size=3200)]
    flags, _, _, thr = _run(cfg, scores, protocol="self_label")
    assert flags[-1000:].mean() > 0.8            # the alarm storm
    assert thr.alpha_t > cfg.threshold.alpha * 3  # alpha_t ran away upward


def test_self_labelled_tau_is_a_non_increasing_ratchet():
    """Pins the THEOREM: buffer subset {x : x <= tau} implies
    Q_{1-alpha}(buffer) <= tau for every alpha, so tau can never move up.
    Upward drift is therefore structurally untrackable without exploration."""
    cfg = Config()
    cfg.threshold.policy = "aci"
    cfg.threshold.aci_update_on = "admitted"
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(size=800), rng.normal(1.5, 1.0, size=2200)]
    _, taus, _, _ = _run(cfg, scores, protocol="self_label")
    taus = taus[200:]
    assert np.max(np.diff(taus)) <= 1e-9, "tau moved UP under self-labelling"


def test_both_fixes_are_required_to_recover():
    """The mitigation claim, stated as a 2x2. Either fix alone is insufficient;
    only their combination restores nominal FPR. If a future change makes one
    of them sufficient, this test fails and the paper's table is wrong."""
    rng_scores = np.random.default_rng(0)
    scores = np.r_[rng_scores.normal(size=800), rng_scores.normal(1.0, 1.0, size=3200)]
    out = {}
    for mode in ["admitted", "all"]:
        for ex in [0.0, 0.10]:
            cfg = Config()
            cfg.threshold.policy = "aci"
            cfg.threshold.gamma_aci = 0.01
            cfg.threshold.aci_update_on = mode
            flags, _, _, _ = _run(cfg, scores, protocol="self_label", explore=ex)
            out[(mode, ex)] = flags[-1000:].mean()
    assert out[("admitted", 0.0)] > 0.5
    assert out[("admitted", 0.10)] > 0.12
    assert out[("all", 0.0)] > 0.12
    assert out[("all", 0.10)] < 0.12, out


def test_alpha_is_clamped_to_an_attainable_level():
    """Without the clamp, alpha_t drifts below 1/(n+1), the quantile returns
    +inf, and the detector switches off while its FPR log reads 0.000 -- the
    most dangerous possible failure, because it looks like success."""
    cfg = Config(); cfg.threshold.policy = "aci"; cfg.threshold.window = 40
    rng = np.random.default_rng(0)
    _, taus, _, thr = _run(cfg, np.r_[rng.normal(size=500), rng.normal(-4, 1, size=2000)])
    assert np.isfinite(thr.tau)
    assert thr.alpha_t >= 1.0 / (len(thr.buffer) + 1)


def test_state_dict_round_trip_preserves_behaviour():
    """Resume must not perturb the threshold. If it does, every run that
    crossed a Kaggle session boundary is a different experiment from one that
    did not, and the two cannot be pooled."""
    cfg = Config(); cfg.threshold.policy = "aci"
    rng = np.random.default_rng(0)
    s = rng.normal(size=600)
    a = build_threshold(cfg.threshold, rng.normal(size=200))
    for x in s[:300]:
        a.observe(x, True, flagged=a.flag(x))
    b = build_threshold(cfg.threshold, rng.normal(size=200))
    b.load_state_dict(a.state_dict())
    for x in s[300:]:
        fa, fb = a.flag(x), b.flag(x)
        assert fa == fb
        a.observe(x, True, flagged=fa); b.observe(x, True, flagged=fb)
    assert a.tau == pytest.approx(b.tau)
