"""What does each feedback scheme actually control?

WHY THIS FILE EXISTS

Round-3 review found that the repair the paper proposed (P1: feed ACI the alarm
on every sample) does not regulate nominal FPR at all. ACI drives the long-run
mean of whatever indicator it is fed toward alpha. Feed it every sample's alarm
and it regulates the total alarm rate (1-pi)FPR + pi*TPR, so nominal FPR settles
at (alpha - pi*TPR)/(1-pi). That one identity explains the "unresolved" four-fold
conservatism, shows the "oracle" comparator was never label-supervised, and
predicts the detector goes blind during defect bursts.

Every test below reproduces a claim that was verified by simulation before being
written down. Several pin results that contradict what the paper originally
said, so that the corrected claims cannot silently regress.
"""
import numpy as np
import pytest

from cgc_ad.config import Config
from cgc_ad.conformal import build_threshold, controlled_quantity

ALPHA, PI, MU_A = 0.05, 0.05, 2.17        # MU_A gives TPR ~0.70 near tau ~1.645


def _run(protocol, update_on, explore=0.0, T=12000, shift=None, burst=None,
         seed=0, gamma=0.01):
    rng = np.random.default_rng(seed)
    c = Config()
    c.threshold.policy = "aci"
    c.threshold.gamma_aci = gamma
    c.threshold.aci_update_on = update_on
    c.threshold.window = 200
    thr = build_threshold(c.threshold, rng.normal(size=200))
    lab, flg, tau = [], [], []
    for t in range(T):
        p = burst[2] if burst and burst[0] <= t < burst[1] else PI
        y = int(rng.random() < p)
        sh = 0.0
        if shift:
            sh = min(1.0, max(0.0, (t - shift[0]) / shift[1])) * shift[2]
        s = rng.normal(MU_A if y else 0.0) + sh
        f = thr.flag(s)
        if protocol == "clean":
            adm = (y == 0)
        elif protocol == "self":
            adm = (not f) or (rng.random() < explore)
        elif protocol == "reviewed":
            adm = (not f) or (y == 0)
        else:
            raise ValueError(protocol)
        thr.observe(s, adm, flagged=f)
        lab.append(y); flg.append(f); tau.append(thr.tau)
    return np.array(lab), np.array(flg, dtype=float), np.array(tau)


def _rates(lab, flg, a, b):
    L, F = lab[a:b], flg[a:b]
    return F[L == 0].mean(), F[L == 1].mean(), F.mean()


# ------------------------------------------------------------ identifiability

def test_all_sample_feedback_regulates_the_alarm_rate_not_fpr():
    """THE CENTRAL CORRECTION. With err on every sample, the total alarm rate
    converges to alpha and nominal FPR to (alpha - pi*TPR)/(1-pi). This is why
    the repaired system sat at ~0.013-0.02 instead of 0.05."""
    lab, flg, _ = _run("clean", "all")
    fpr, tpr, alarm = _rates(lab, flg, 3000, 12000)
    assert alarm == pytest.approx(ALPHA, abs=0.006)
    predicted = (ALPHA - PI * tpr) / (1 - PI)
    assert fpr == pytest.approx(predicted, abs=0.006), (fpr, predicted)
    assert fpr < 0.035, "alarm-rate control must sit well below nominal alpha"


def test_nominal_only_feedback_is_genuinely_labelled_aci():
    """The baseline the paper called 'oracle' should have been this arm. Error
    restricted to known-nominal samples holds nominal FPR at alpha."""
    lab, flg, _ = _run("clean", "admitted")
    fpr, _, _ = _rates(lab, flg, 3000, 12000)
    assert fpr == pytest.approx(ALPHA, abs=0.006)


def test_controlled_quantity_labels_every_arm_correctly():
    """Mechanically prevents an arm being reported as label-supervised when its
    feedback stream regulates something else."""
    assert "nominal FPR" in controlled_quantity("clean", "admitted")
    assert "ALARM RATE" in controlled_quantity("clean", "all")
    assert "ALARM RATE" in controlled_quantity("self_label", "all")
    assert "DEGENERATE" in controlled_quantity("self_label", "admitted")
    assert "one-sided" in controlled_quantity("reviewed", "admitted")


# ------------------------------------------------------ the monotonicity lemma

def test_naive_threshold_is_non_increasing_with_no_drift_at_all():
    """LEMMA. Under score-thresholded admission with admission-only feedback,
    alpha_t is non-decreasing (Prop. 1), so k_t is non-increasing, and each
    admission adds an element <= tau_{t-1}. Hence tau_t <= tau_{t-1} from t=1
    onward, with or without drift. This is why the collapse appears BEFORE
    drift onset: it is not a drift phenomenon."""
    lab, flg, tau = _run("self", "admitted", T=6000, shift=None)
    assert np.max(np.diff(tau[1:])) <= 1e-12
    fpr, _, _ = _rates(lab, flg, 4000, 6000)
    assert fpr > 0.8, "naive self-labelling must collapse without any drift"


def test_buffer_maximum_is_non_increasing_under_any_score_thresholded_admission():
    """The corrected Proposition 2, and the reason it also binds P1: with any
    controller, if admission requires s <= tau_{t-1} <= max B_{t-1}, then max B_t
    can only fall. P1 changes the controller, not the admission rule."""
    from cgc_ad.conformal import build_threshold
    rng = np.random.default_rng(0)
    c = Config(); c.threshold.policy = "aci"; c.threshold.aci_update_on = "all"
    thr = build_threshold(c.threshold, rng.normal(size=200))
    maxes = []
    for t in range(4000):
        s = rng.normal(1.0 if t > 1000 else 0.0)       # upward drift
        f = thr.flag(s)
        thr.observe(s, not f, flagged=f)               # self-labelled, P1 feedback
        maxes.append(max(thr.buffer))
    assert np.max(np.diff(maxes)) <= 1e-12


def test_exploration_lets_the_buffer_maximum_rise():
    """P2 is what breaks the ceiling -- but only softly: explored samples enter
    at rate rho, so the tail is under-represented by that factor."""
    rng = np.random.default_rng(1)
    c = Config(); c.threshold.policy = "aci"; c.threshold.aci_update_on = "all"
    thr = build_threshold(c.threshold, rng.normal(size=200))
    m0 = max(thr.buffer)
    for t in range(4000):
        s = rng.normal(2.0 if t > 500 else 0.0)
        f = thr.flag(s)
        thr.observe(s, (not f) or rng.random() < 0.05, flagged=f)
    assert max(thr.buffer) > m0


# ------------------------------------------- table II row 2 is not open-loop

def test_explored_errors_close_the_loop_around_a_biased_estimate():
    """'Admitted only, rho=0.05' was described as open-loop. It is not: explored
    flagged samples enter with err=1. The observed error rate among admitted
    samples is rho*a/((1-a)+rho*a); ACI drives it to alpha, so the equilibrium
    alarm rate is a* = alpha/(alpha + rho(1-alpha)) ~ 0.513. The paper's 0.4671
    is this equilibrium, not an open-loop drift."""
    lab, flg, _ = _run("self", "admitted", explore=0.05)
    _, _, alarm = _rates(lab, flg, 3000, 12000)
    a_star = ALPHA / (ALPHA + 0.05 * (1 - ALPHA))
    assert alarm == pytest.approx(a_star, abs=0.03)


# ------------------------------------------------------ the burst failure mode

def test_alarm_rate_control_goes_blind_during_a_defect_burst():
    """THE PRACTICAL DANGER. When defects cluster (tool wear, a bad lot), alarms
    rise, alarm-rate control pushes alpha_t down and tau up, and the detector
    desensitises exactly when it is most needed."""
    lab, flg, _ = _run("clean", "all", burst=(6000, 6600, 0.40))
    tpr_before = flg[4000:6000][lab[4000:6000] == 1].mean()
    tpr_burst = flg[6200:6600][lab[6200:6600] == 1].mean()
    assert tpr_burst < tpr_before - 0.15, (tpr_before, tpr_burst)


def test_the_proposed_patch_is_also_blinded_by_bursts():
    """The self-label + P1 + P2 method the paper proposed inherits the same
    failure, and in simulation is the worst non-degenerate arm under bursts."""
    lab, flg, _ = _run("self", "all", explore=0.05, burst=(6000, 6600, 0.40))
    tpr_burst = flg[6200:6600][lab[6200:6600] == 1].mean()
    assert tpr_burst < 0.35, tpr_burst


def test_labelled_feedback_is_burst_robust():
    lab, flg, _ = _run("clean", "admitted", burst=(6000, 6600, 0.40))
    tpr_before = flg[4000:6000][lab[4000:6000] == 1].mean()
    tpr_burst = flg[6200:6600][lab[6200:6600] == 1].mean()
    assert tpr_burst >= tpr_before - 0.05


# ------------------------------------------------------ the constructive repair

@pytest.mark.parametrize("regime", ["stationary", "drift", "burst"])
def test_reviewed_alarms_match_full_labels(regime):
    """THE REPAIR THE PAPER SHOULD PROPOSE. Inspection lines already route
    flagged parts to human review. False alarms are exactly the upper tail of the
    nominal distribution that self-labelling censors, so re-admitting reviewed
    false alarms un-censors the buffer, and excluding reviewed true defects from
    the update prevents burst desensitisation. Labels are needed on flagged parts
    only (~8% of the stream here). Predicted FPR = alpha(1-pi*TPR)/(1-pi) ~ 0.0508."""
    kw = {"stationary": {}, "drift": {"shift": (3000, 1000, 1.5)},
          "burst": {"burst": (6000, 6600, 0.40)}}[regime]
    win = {"stationary": (3000, 12000), "drift": (5000, 12000),
           "burst": (6200, 6600)}[regime]
    lab_r, flg_r, _ = _run("reviewed", "admitted", **kw)
    lab_l, flg_l, _ = _run("clean", "admitted", **kw)
    fpr_r, tpr_r, _ = _rates(lab_r, flg_r, *win)
    fpr_l, tpr_l, _ = _rates(lab_l, flg_l, *win)
    assert abs(fpr_r - fpr_l) < 0.012, (regime, fpr_r, fpr_l)
    assert abs(tpr_r - tpr_l) < 0.06, (regime, tpr_r, tpr_l)
    if regime != "burst":
        assert fpr_r == pytest.approx(ALPHA, abs=0.008)
