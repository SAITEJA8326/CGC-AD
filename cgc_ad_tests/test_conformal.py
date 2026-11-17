"""Conformal machinery.

WHY THESE TESTS EXIST
The conformal quantile is the one place in the system where an off-by-one is
both invisible and consequential: a wrong order statistic still returns a
plausible number, still produces a plausible FPR, and quietly voids the only
distribution-free guarantee the paper claims. Everything downstream (the
curriculum error signal, every threshold policy, the p-value stream feeding the
martingale) is a function of this number.
"""
import numpy as np
import pytest
from cgc_ad.conformal import (conformal_quantile, trimmed_conformal_quantile,
                              conformal_pvalue, PowerMartingale)


def test_order_statistic_is_exact():
    """The threshold must be the ceil((n+1)(1-alpha))-th smallest score, not
    np.quantile. They differ by one rank, which is the entire finite-sample
    correction."""
    s = np.arange(1, 101, dtype=float)                    # n = 100
    assert conformal_quantile(s, 0.05) == 96.0            # ceil(101*0.95) = 96
    assert conformal_quantile(s, 0.10) == 91.0            # ceil(101*0.90) = 91
    assert conformal_quantile(s, 0.50) == 51.0


def test_unattainable_level_returns_inf_not_a_plausible_number():
    """With n < 1/alpha - 1 there is NO finite threshold at that level. Silently
    returning the max would look fine and be wrong; +inf forces the caller to
    notice. This is the toothbrush failure (n_calib=15, alpha=0.05)."""
    s = np.arange(1, 16, dtype=float)
    assert conformal_quantile(s, 0.05) == float("inf")
    assert np.isfinite(conformal_quantile(s, 0.10))
    assert conformal_quantile(np.array([]), 0.05) == float("inf")


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_coverage_is_at_least_1_minus_alpha(alpha):
    """The property the whole framework rests on. Empirical coverage over many
    independent calibration/test draws must be >= 1 - alpha (conservative by
    construction), and not much above it."""
    rng = np.random.default_rng(0)
    cov = []
    for _ in range(4000):
        cal = rng.normal(size=200)
        cov.append(rng.normal() <= conformal_quantile(cal, alpha))
    c = float(np.mean(cov))
    assert 1 - alpha - 0.02 <= c <= 1 - alpha + 0.03, c


def test_coverage_holds_for_arbitrary_score_distribution():
    """Distribution-free means distribution-free. A heavy-tailed, skewed score
    must give the same coverage as a Gaussian one; if it does not, something is
    assuming normality."""
    rng = np.random.default_rng(1)
    cov = []
    for _ in range(3000):
        cal = rng.standard_cauchy(size=200)
        cov.append(rng.standard_cauchy() <= conformal_quantile(cal, 0.10))
    assert 0.87 <= float(np.mean(cov)) <= 0.93


def test_trimming_lowers_the_threshold_and_removes_upper_outliers():
    """Trimming must discard the contaminated upper tail. If it raised the
    threshold instead, robust_aci would amplify contamination rather than
    resist it."""
    rng = np.random.default_rng(2)
    clean = rng.normal(size=190)
    dirty = np.r_[clean, rng.normal(6, 1, size=10)]
    assert trimmed_conformal_quantile(dirty, 0.05, 0.05) < conformal_quantile(dirty, 0.05)
    assert trimmed_conformal_quantile(dirty, 0.05, 0.05) < dirty.max()


def test_pvalue_is_uniform_under_exchangeability():
    """The martingale is only valid if the p-value stream is uniform when
    nothing has changed. If p is biased, the drift detector fires on nothing."""
    rng = np.random.default_rng(3)
    cal = rng.normal(size=500)
    ps = np.array([conformal_pvalue(rng.normal(), cal) for _ in range(4000)])
    assert 0.45 < ps.mean() < 0.55
    assert abs(np.mean(ps < 0.1) - 0.1) < 0.02


def test_martingale_does_not_fire_on_uniform_but_does_on_skew():
    """Ville's inequality bounds the false-alarm probability at 1/lambda. A
    trigger that fires under exchangeability would make every adaptation event
    spurious."""
    rng = np.random.default_rng(4)
    m = PowerMartingale(lam=20.0)
    assert sum(m.update(rng.uniform()) for _ in range(5000)) <= 1
    m2 = PowerMartingale(lam=20.0)
    assert sum(m2.update(rng.uniform() ** 4) for _ in range(2000)) > 10
