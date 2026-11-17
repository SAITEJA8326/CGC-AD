"""Controllers, the hardness axis, and the control signal.

WHY THESE TESTS EXIST
The training-time controller was refuted empirically (random hardness beat it by
4.7 AUROC points). These tests pin down WHY, so the negative result is a
characterised mechanism rather than an anecdote, and so the ablation arms the
paper reports keep behaving the way the paper says they do.
"""
import numpy as np
import pytest
from cgc_ad.config import Config
from cgc_ad.curriculum import build_controller, miss_rate, _norm_ppf

KINDS = ["fixed", "random", "oracle", "proportional", "pid", "distributional"]


@pytest.mark.parametrize("kind", KINDS)
def test_hardness_stays_in_range(kind, rng):
    """h parameterises the synthesiser; out-of-range values would silently clamp
    inside synth.py and make the logged h a lie."""
    cfg = Config(); cfg.curriculum.kind = kind
    c = build_controller(cfg.curriculum, rng, 6)
    for _ in range(200):
        c.update(float(rng.uniform()), rng.uniform(size=6))
        assert 0.0 <= c.hardness() <= 1.0
        assert 0.0 <= c.sample_hardness() <= 1.0


def test_setpoint_controller_collapses_the_support():
    """THE MECHANISM behind the negative result. A proportional controller
    converges to a point, so the model sees a narrow band of hardness. Random
    keeps the full range. Diversity is what the empirical result says matters,
    and this test shows the controller destroys it by construction."""
    cfg = Config(); cfg.curriculum.kind = "proportional"
    p = build_controller(cfg.curriculum, np.random.default_rng(0), 6)
    r = build_controller(Config(**{}).curriculum.__class__(kind="random"),
                         np.random.default_rng(0), 6)
    hp, hr = [], []
    for _ in range(300):
        p.update(0.35); r.update(0.35)         # constant miss above target
        hp.append(p.sample_hardness()); hr.append(r.sample_hardness())
    assert np.std(hp[-100:]) < 0.02, "proportional should collapse to a point"
    assert np.std(hr[-100:]) > 0.20, "random should retain spread"


def test_proportional_moves_hardness_in_the_correct_direction():
    """Sign errors in a feedback loop are self-concealing: the system still
    converges, just to the wrong end of the axis."""
    cfg = Config(); cfg.curriculum.kind = "proportional"
    c = build_controller(cfg.curriculum, np.random.default_rng(0), 6)
    h0 = c.hardness()
    for _ in range(20):
        c.update(cfg.curriculum.target_miss + 0.3)   # too many easy misses
    assert c.hardness() > h0                          # -> harder
    c2 = build_controller(cfg.curriculum, np.random.default_rng(0), 6)
    for _ in range(20):
        c2.update(0.0)                                # nothing is missed
    assert c2.hardness() < h0                         # -> easier


def test_deadband_suppresses_noise_driven_drift():
    """The miss-rate estimate carries SE ~ sqrt(g(1-g)/m). Without a deadband
    the controller random-walks on estimation noise alone."""
    cfg = Config(); cfg.curriculum.kind = "proportional"; cfg.curriculum.deadband = 0.05
    c = build_controller(cfg.curriculum, np.random.default_rng(0), 6)
    h0 = c.hardness()
    rng = np.random.default_rng(1)
    for _ in range(500):
        c.update(cfg.curriculum.target_miss + rng.normal(0, 0.01))
    assert abs(c.hardness() - h0) < 1e-9


def test_distributional_controller_keeps_entropy():
    """The proposed replacement must not collapse the way the setpoint does;
    that is its entire justification."""
    cfg = Config(); cfg.curriculum.kind = "distributional"
    c = build_controller(cfg.curriculum, np.random.default_rng(0), 6)
    per = np.linspace(0.02, 0.6, 6)
    for _ in range(100):
        c.update(0.2, per)
    assert np.std([c.sample_hardness() for _ in range(400)]) > 0.10


def test_control_signals_agree_on_gaussian_scores():
    """THE F2 ABLATION, as a test. If `conformal`, `zscore` and
    `synth_percentile` return the same miss-rate on well-behaved scores, the
    conformal reference is buying only scale invariance -- which is a real but
    much smaller claim than 'conformal-guided'."""
    rng = np.random.default_rng(0)
    n = rng.normal(size=400); s = rng.normal(1.5, 1.0, size=400)
    cfg = Config()
    out = {}
    for sig in ["conformal", "zscore", "synth_percentile"]:
        cfg.curriculum.signal = sig
        out[sig] = miss_rate(s, n, cfg.curriculum, 0.05)
    assert abs(out["zscore"] - out["synth_percentile"]) < 0.05
    assert abs(out["conformal"] - out["synth_percentile"]) < 0.10, out


def test_conformal_signal_is_invariant_to_monotone_rescaling():
    """The ONE property the conformal reference genuinely provides. Raw-threshold
    signals break here, which is the honest argument for keeping it."""
    rng = np.random.default_rng(0)
    n = rng.normal(size=400); s = rng.normal(1.5, 1.0, size=400)
    cfg = Config(); cfg.curriculum.signal = "conformal"
    a = miss_rate(s, n, cfg.curriculum, 0.05)
    b = miss_rate(3 * s + 7, 3 * n + 7, cfg.curriculum, 0.05)
    assert a == pytest.approx(b)
    cfg.curriculum.signal = "raw"; cfg.curriculum.raw_threshold = 1.0
    assert miss_rate(s, n, cfg.curriculum, 0.05) != \
           pytest.approx(miss_rate(3 * s + 7, 3 * n + 7, cfg.curriculum, 0.05))


def test_norm_ppf_matches_known_quantiles():
    """A wrong constant here silently rescales the zscore ablation arm; the
    first version of this code used 2.33 where 1.645 was meant."""
    assert _norm_ppf(0.95) == pytest.approx(1.6448536, abs=1e-5)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)


# ==========================================================================
# Support collapse -- restored coverage for the probability floor
# ==========================================================================

def test_distributional_collapses_without_the_probability_floor():
    """WHY: support collapse is exactly what made the setpoint controller lose
    to random hardness. A categorical controller that drives rungs to zero has
    the same defect in a different shape. `min_prob` mixes with the uniform so
    no hardness level can be extinguished; this test keeps the defect
    documented and executable so the floor cannot be silently removed."""
    from cgc_ad.config import CurriculumConfig
    per = np.array([0.5, 0.5, 0.5, 0.15, 0.5, 0.5])   # only one rung on target
    ents = {}
    for floor in [0.0, 0.05]:
        c = build_controller(
            CurriculumConfig(kind="distributional", min_prob=floor,
                             temperature=0.02, rho=0.6),
            np.random.default_rng(0), 6)
        for _ in range(200):
            c.update(0.2, per)
        ents[floor] = float(-(c.p * np.log(c.p + 1e-12)).sum())
    assert ents[0.0] < 0.15, "without a floor the distribution should collapse"
    assert ents[0.05] > 0.5, "the floor must preserve support"


def test_distributional_still_prefers_the_on_target_rung():
    """The floor must not neuter the controller: with support preserved, the
    rung nearest the target miss-rate should still carry the most mass."""
    from cgc_ad.config import CurriculumConfig
    per = np.array([0.5, 0.5, 0.5, 0.15, 0.5, 0.5])
    c = build_controller(CurriculumConfig(kind="distributional", min_prob=0.05,
                                          temperature=0.05, rho=0.4),
                         np.random.default_rng(0), 6)
    for _ in range(100):
        c.update(0.2, per)
    assert int(np.argmax(c.p)) == 3


def test_controller_state_round_trips():
    """WHY: adaptation events happen mid-stream and are checkpointed. A
    controller whose state does not survive resume makes a session-interrupted
    run a different experiment from an uninterrupted one."""
    from cgc_ad.config import CurriculumConfig
    for kind in KINDS:
        a = build_controller(CurriculumConfig(kind=kind),
                             np.random.default_rng(0), 6)
        for _ in range(30):
            a.update(0.3, np.linspace(0.05, 0.6, 6))
        b = build_controller(CurriculumConfig(kind=kind),
                             np.random.default_rng(0), 6)
        b.load_state_dict(a.state_dict())
        assert b.hardness() == pytest.approx(a.hardness()), kind


def test_oracle_controller_holds_the_value_it_is_given():
    """The oracle arm is the upper bound the controller is trying to reach. If
    it drifted under updates it would not be an upper bound on anything."""
    from cgc_ad.config import CurriculumConfig
    c = build_controller(CurriculumConfig(kind="oracle"),
                         np.random.default_rng(0), 6)
    c.set_oracle(0.8)
    for _ in range(50):
        c.update(0.9, np.linspace(0.05, 0.6, 6))
    assert c.hardness() == pytest.approx(0.8)
