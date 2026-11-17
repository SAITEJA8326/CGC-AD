"""Conformal thresholds, online threshold policies, and drift evidence.

Four threshold policies share one interface so the main table is a loop:

    static      fixed tau from the initial calibration set (the deployed default)
    window      recompute the conformal quantile over the last W buffer entries
                -- the strongest and simplest competitor to ACI (review F8)
    aci         Gibbs & Candes adaptive conformal inference on alpha_t
    robust_aci  ACI with a contamination-corrected / trimmed quantile (review F5)

`window` is in the main table on purpose. If it ties with `aci`, the ACI
component is not a contribution and the paper must say so.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------
# Split-conformal quantile
# --------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th order statistic of the calibration scores.

    Returns +inf when the level is unattainable at this sample size, i.e. when
    n < 1/alpha - 1. That is not a bug: it is the honest answer, and it is why
    the review insists on reporting calibration-set size (F6).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def trimmed_conformal_quantile(scores: np.ndarray, alpha: float,
                               eps: float) -> float:
    """Contamination-corrected quantile.

    If at most an eps fraction of the buffer are true anomalies (which sit in
    the upper tail), the (1-alpha) quantile of the *clean* distribution is at
    most the (1-alpha)(1-eps)+eps ... quantile of the observed one. We take the
    conservative-but-simple route: drop the top eps fraction, then conformalise
    on the remainder.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n == 0:
        return float("inf")
    keep = max(1, int(round(n * (1.0 - max(0.0, eps)))))
    return conformal_quantile(s[:keep], alpha)


def conformal_pvalue(score: float, calib: np.ndarray) -> float:
    """Smoothed-free conformal p-value: P(score >= calibration scores)."""
    n = calib.size
    if n == 0:
        return 1.0
    return float((1 + np.sum(calib >= score)) / (n + 1))


# --------------------------------------------------------------------------
# Threshold policies
# --------------------------------------------------------------------------

class BaseThreshold:
    name = "base"

    def __init__(self, cfg, calib_scores: np.ndarray):
        self.cfg = cfg
        self.alpha_target = cfg.alpha
        self.buffer: Deque[float] = deque(maxlen=cfg.window)
        for s in calib_scores[-cfg.window:]:
            self.buffer.append(float(s))
        self.tau = conformal_quantile(np.asarray(calib_scores), cfg.alpha)
        self.alpha_t = cfg.alpha

    def flag(self, score: float) -> bool:
        return score > self.tau

    def observe(self, score: float, admitted: bool,
                flagged: Optional[bool] = None) -> None:
        """Update from one observation.

        `admitted` = the sample entered the presumed-normal buffer (governs the
        quantile estimate). `flagged` = the alarm decision actually taken
        (governs the ACI error indicator). Keeping these separate is what
        prevents the self-labelling degeneracy documented in ThresholdConfig.
        """
        raise NotImplementedError

    def _alarm_indicator(self, score: float, admitted: bool,
                         flagged: Optional[bool]) -> Optional[float]:
        mode = getattr(self.cfg, "aci_update_on", "all")
        if mode == "admitted":
            return None if not admitted else (1.0 if score > self.tau else 0.0)
        f = (score > self.tau) if flagged is None else bool(flagged)
        return 1.0 if f else 0.0

    def _attainable_alpha(self, alpha: float, n: int) -> float:
        """Clamp alpha to a level the buffer can actually express.

        With n calibration points the smallest attainable level is 1/(n+1);
        below it `conformal_quantile` returns +inf and the detector stops
        flagging anything while its FPR log still looks perfect. Clamping
        makes that failure visible instead of silent.
        """
        return float(np.clip(alpha, 1.5 / (n + 1), 0.5))

    def _rate_limit(self, new_tau: float) -> float:
        ms = getattr(self.cfg, "max_step", None)
        if ms is None or not math.isfinite(self.tau):
            return new_tau
        return float(np.clip(new_tau, self.tau - ms, self.tau + ms))

    def state(self) -> Dict[str, float]:
        return {"tau": self.tau, "alpha_t": self.alpha_t,
                "buffer": len(self.buffer)}

    # ---- checkpointing ---------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {"tau": self.tau, "alpha_t": self.alpha_t,
                "alpha_raw": getattr(self, "alpha_raw", self.alpha_t),
                "alpha_target": self.alpha_target,
                "buffer": list(self.buffer)}

    def load_state_dict(self, st: Dict[str, object]) -> None:
        self.tau = float(st["tau"])
        self.alpha_t = float(st["alpha_t"])
        self.alpha_raw = float(st.get("alpha_raw", st["alpha_t"]))
        self.alpha_target = float(st["alpha_target"])
        self.buffer = deque(st["buffer"], maxlen=self.cfg.window)


class StaticThreshold(BaseThreshold):
    name = "static"

    def observe(self, score: float, admitted: bool,
                flagged: Optional[bool] = None) -> None:
        return


class WindowThreshold(BaseThreshold):
    """Recompute the split-conformal quantile over the rolling buffer."""
    name = "window"

    def observe(self, score: float, admitted: bool,
                flagged: Optional[bool] = None) -> None:
        if admitted:
            self.buffer.append(float(score))
        if len(self.buffer) >= 20:
            self.tau = self._rate_limit(
                conformal_quantile(
                    np.fromiter(self.buffer, dtype=np.float64),
                    self._attainable_alpha(self.alpha_target, len(self.buffer))))


class ACIThreshold(BaseThreshold):
    """Gibbs & Candes (2021) adaptive conformal inference.

        alpha_{t+1} = alpha_t + gamma * (alpha - 1[s_t > tau_t])
        tau_t       = Q_{1-alpha_t}(buffer)

    Two streams feed this, and they are deliberately NOT the same one:

      * the **quantile buffer** takes admitted (presumed-normal) samples;
      * the **error indicator** takes the alarm decision on every observed
        sample when `aci_update_on="all"`.

    Coupling them (`aci_update_on="admitted"`) is degenerate under a
    self-labelled buffer: admission is *defined* as `not flagged`, so the
    indicator is identically zero, alpha_t ramps monotonically to its cap at
    gamma*alpha per step, and tau collapses toward the median of an
    increasingly left-censored buffer. That is a structural failure of the
    update rule, not a tuning problem.

    Note also that the anytime coverage guarantee applies to the *observed*
    stream; with a self-labelled buffer that stream is not the normal
    distribution, so the guarantee does not transfer. See review F5.
    """
    name = "aci"

    def _aci_step(self, err: float) -> None:
        self.alpha_raw = float(np.clip(
            getattr(self, "alpha_raw", self.alpha_t)
            + self.cfg.gamma_aci * (self.alpha_target - err), 1e-4, 0.5))

    def _refit(self) -> None:
        self.alpha_t = self._attainable_alpha(self.alpha_raw, len(self.buffer))
        self.tau = self._rate_limit(
            conformal_quantile(np.fromiter(self.buffer, dtype=np.float64),
                               self.alpha_t))

    def observe(self, score: float, admitted: bool,
                flagged: Optional[bool] = None) -> None:
        err = self._alarm_indicator(score, admitted, flagged)
        if err is not None:
            self._aci_step(err)
        if not admitted:
            return
        self.buffer.append(float(score))
        if len(self.buffer) >= 20:
            self._refit()


class RobustACIThreshold(ACIThreshold):
    """ACI on a trimmed quantile, for contaminated buffers.

    Only expected to pay off when the buffer actually is contaminated; on a
    clean buffer the trim biases tau downward and costs calibration.
    """
    name = "robust_aci"

    def _refit(self) -> None:
        n_eff = int(len(self.buffer) * (1.0 - max(0.0, self.cfg.trim_eps)))
        self.alpha_t = self._attainable_alpha(self.alpha_raw, max(1, n_eff))
        self.tau = self._rate_limit(
            trimmed_conformal_quantile(
                np.fromiter(self.buffer, dtype=np.float64),
                self.alpha_t, self.cfg.trim_eps))


def controlled_quantity(buffer_protocol: str, aci_update_on: str) -> str:
    """What a (buffer protocol, feedback stream) pair actually regulates.

    ACI drives the long-run mean of whatever indicator it is fed toward alpha
    (Gibbs & Candes, telescoping identity). Feeding it the alarm on EVERY
    sample regulates the total alarm rate (1-pi)*FPR + pi*TPR, not the nominal
    FPR. Nominal FPR is only regulated when the feedback is restricted to
    samples known to be nominal. This function exists so that an arm can
    never again be labelled "oracle" or "label-supervised" while its feedback
    stream says otherwise.
    """
    if buffer_protocol == "clean" and aci_update_on == "admitted":
        return "nominal FPR (labelled ACI)"
    if buffer_protocol == "reviewed" and aci_update_on == "admitted":
        return "nominal FPR, approx. alpha(1-pi*TPR)/(1-pi) (one-sided labels)"
    if aci_update_on == "all":
        return "ALARM RATE (1-pi)FPR + pi*TPR -- not nominal FPR"
    if buffer_protocol in ("self_label", "gated") and aci_update_on == "admitted":
        return "DEGENERATE: feedback identically zero (Prop. 1)"
    return "unknown"


_POLICIES = {
    "static": StaticThreshold, "window": WindowThreshold,
    "aci": ACIThreshold, "robust_aci": RobustACIThreshold,
}


def build_threshold(cfg, calib_scores: np.ndarray) -> BaseThreshold:
    if cfg.policy not in _POLICIES:
        raise ValueError(f"unknown threshold policy {cfg.policy}; "
                         f"choose from {sorted(_POLICIES)}")
    return _POLICIES[cfg.policy](cfg, calib_scores)


# --------------------------------------------------------------------------
# Drift evidence: conformal test martingale (Vovk power martingale)
# --------------------------------------------------------------------------

class PowerMartingale:
    """Accumulates evidence that the p-value stream has stopped being uniform.

    Under exchangeability the conformal p-values are (approximately) uniform
    and M_t is a nonnegative martingale started at 1, so Ville's inequality
    gives P(sup_t M_t >= lambda) <= 1/lambda. Crossing lambda is therefore an
    alarm with a controlled false-trigger rate -- which is what makes this a
    principled adaptation trigger rather than another tuned heuristic.
    """

    def __init__(self, epsilon: float = 0.6, lam: float = 20.0):
        self.eps = epsilon
        self.lam = lam
        self.log_m = 0.0
        self.n_alarms = 0

    @property
    def value(self) -> float:
        return float(np.exp(min(self.log_m, 700.0)))

    def update(self, p: float) -> bool:
        p = float(np.clip(p, 1e-6, 1.0))
        self.log_m += math.log(self.eps) + (self.eps - 1.0) * math.log(p)
        self.log_m = max(self.log_m, -20.0)          # keep it from underflowing
        if self.log_m >= math.log(self.lam):
            self.n_alarms += 1
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.log_m = 0.0

    def state_dict(self) -> Dict[str, float]:
        return {"log_m": self.log_m, "n_alarms": self.n_alarms,
                "eps": self.eps, "lam": self.lam}

    def load_state_dict(self, st: Dict[str, float]) -> None:
        self.log_m = float(st["log_m"])
        self.n_alarms = int(st["n_alarms"])
