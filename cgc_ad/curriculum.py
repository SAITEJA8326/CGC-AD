"""Hardness controllers for the synthetic-anomaly curriculum.

Six arms, all behind one interface:

    fixed           h = h0                       (CutPaste/NSA-style baseline)
    random          h ~ U[0,1] each epoch        <- THE decisive control (F8)
    oracle          h = best fixed h by grid search on the validation split
    proportional    the original CGC-AD controller
    pid             proportional + integral + derivative
    distributional  a categorical distribution over the hardness ladder (3.3)

`random` matters more than any other arm in the project. If it matches the
controller, the benefit is hardness *diversity*, not calibration-driven
targeting, and the central claim is dead. Run it in week 8, not month 4.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class BaseController:
    name = "base"

    def __init__(self, cfg, rng: np.random.Generator, n_levels: int):
        self.cfg = cfg
        self.rng = rng
        self.n_levels = n_levels
        self.h = float(np.clip(cfg.h0, 0.0, 1.0))
        self.history: List[Dict[str, float]] = []
        self._ema: Optional[float] = None
    # ---- state management ------------------------------------------------
    def state_dict(self) -> dict:
        """Returns the controller's state for checkpointing."""
        return self.__dict__.copy()

    def load_state_dict(self, state: dict) -> None:
        """Restores the controller's state from a checkpoint."""
        self.__dict__.update(state)

    # ---- interface -------------------------------------------------------
    def hardness(self) -> float:
        """Hardness for the next epoch (scalar arms)."""
        return self.h

    def sample_hardness(self) -> float:
        """Hardness for a single synthetic sample. Scalar arms just reuse h."""
        return self.hardness()

    def update(self, miss_rate: float, per_level_miss: Optional[np.ndarray] = None
               ) -> None:
        raise NotImplementedError

    # ---- shared helpers --------------------------------------------------
    def _smooth(self, x: float) -> float:
        a = self.cfg.ema
        self._ema = x if self._ema is None else a * x + (1 - a) * self._ema
        return self._ema

    def _log(self, **kw) -> None:
        self.history.append({"h": self.h, **kw})


class FixedController(BaseController):
    name = "fixed"

    def update(self, miss_rate, per_level_miss=None):
        self._log(miss=miss_rate)


class RandomController(BaseController):
    """Uniform hardness, resampled per epoch. Matched budget, zero feedback."""
    name = "random"

    def update(self, miss_rate, per_level_miss=None):
        self._log(miss=miss_rate)
        self.h = float(self.rng.uniform(0.0, 1.0))

    def sample_hardness(self) -> float:
        return float(self.rng.uniform(0.0, 1.0))


class OracleController(BaseController):
    """Fixed hardness chosen by grid search on the validation split.

    Not a deployable method: it is the upper bound the controller is trying to
    reach without tuning. If oracle ~= proportional, the contribution is
    'saves you a hyperparameter sweep', which is an engineering claim.
    """
    name = "oracle"

    def set_oracle(self, h: float) -> None:
        self.h = float(np.clip(h, 0.0, 1.0))

    def update(self, miss_rate, per_level_miss=None):
        self._log(miss=miss_rate)


class ProportionalController(BaseController):
    """h_{e+1} = h_e + eta * (missrate_e - gamma), with smoothing and deadband.

    The deadband and EMA are not cosmetic: the miss-rate estimate carries a
    standard error of about sqrt(g(1-g)/m) from the finite synthetic sample,
    plus quantile noise from a small calibration set, and the plant is
    non-stationary because the model is still training (review F10).
    """
    name = "proportional"

    def update(self, miss_rate, per_level_miss=None):
        m = self._smooth(float(miss_rate))
        err = m - self.cfg.target_miss
        if abs(err) < self.cfg.deadband:
            err = 0.0
        self.h = float(np.clip(self.h + self.cfg.eta * err, 0.0, 1.0))
        self._log(miss=miss_rate, smoothed=m, err=err)


class PIDController(BaseController):
    name = "pid"

    def __init__(self, cfg, rng, n_levels):
        super().__init__(cfg, rng, n_levels)
        self._int = 0.0
        self._prev = 0.0

    def update(self, miss_rate, per_level_miss=None):
        m = self._smooth(float(miss_rate))
        err = m - self.cfg.target_miss
        if abs(err) < self.cfg.deadband:
            err = 0.0
        self._int = float(np.clip(self._int + err, -5.0, 5.0))
        d = err - self._prev
        self._prev = err
        step = self.cfg.eta * err + self.cfg.ki * self._int + self.cfg.kd * d
        self.h = float(np.clip(self.h + step, 0.0, 1.0))
        self._log(miss=miss_rate, err=err, integral=self._int)


class DistributionalController(BaseController):
    """Categorical distribution over the hardness ladder.

    Reweights toward levels whose current miss-rate sits near the target,
    instead of chasing a scalar setpoint. Monotonicity-free, cannot oscillate,
    and preserves diversity -- see review section 3.3.
    """
    name = "distributional"

    def __init__(self, cfg, rng, n_levels):
        super().__init__(cfg, rng, n_levels)
        self.levels = np.linspace(0.0, 1.0, n_levels)
        self.p = np.ones(n_levels) / n_levels

    def hardness(self) -> float:
        return float(self.levels @ self.p)          # reported value only

    def sample_hardness(self) -> float:
        return float(self.rng.choice(self.levels, p=self.p))

    def update(self, miss_rate, per_level_miss=None):
        if per_level_miss is None:
            self._log(miss=miss_rate)
            return
        w = np.exp(-np.abs(per_level_miss - self.cfg.target_miss)
                   / max(self.cfg.temperature, 1e-6))
        w = w / max(w.sum(), 1e-12)
        self.p = (1 - self.cfg.rho) * self.p + self.cfg.rho * w
        self.p = self.p / self.p.sum()
        self.h = self.hardness()
        self._log(miss=miss_rate, entropy=float(-(self.p * np.log(self.p + 1e-12)).sum()))


_CONTROLLERS = {
    "fixed": FixedController, "random": RandomController,
    "oracle": OracleController, "proportional": ProportionalController,
    "pid": PIDController, "distributional": DistributionalController,
}


def build_controller(cfg, rng: np.random.Generator, n_levels: int) -> BaseController:
    if cfg.kind not in _CONTROLLERS:
        raise ValueError(f"unknown controller {cfg.kind}; "
                         f"choose from {sorted(_CONTROLLERS)}")
    return _CONTROLLERS[cfg.kind](cfg, rng, n_levels)


# --------------------------------------------------------------------------
# Controller error signal (review F2 ablation axis)
# --------------------------------------------------------------------------

def _norm_ppf(q: float) -> float:
    """Standard-normal quantile (Acklam-style rational approximation).

    Avoids a scipy dependency in the hot loop; accurate to ~1e-9 over (0,1).
    """
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        t = math.sqrt(-2 * math.log(q))
        return (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
               ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    if q > ph:
        t = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
                ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    t = q - 0.5
    r = t * t
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def miss_rate(synth_scores: np.ndarray, normal_scores: np.ndarray,
              cfg_curr, alpha: float) -> float:
    """Fraction of synthetic anomalies that fall below the reference threshold.

    The four variants exist to test whether the conformal reference is doing
    anything a cheaper normalisation could not. If `raw`, `zscore`, and
    `synth_percentile` all match `conformal`, drop 'Conformal' from the title.
    """
    from .conformal import conformal_quantile

    s = np.asarray(synth_scores, dtype=np.float64)
    n = np.asarray(normal_scores, dtype=np.float64)
    if s.size == 0:
        return 0.0

    sig = cfg_curr.signal
    if sig == "conformal":
        tau = conformal_quantile(n, alpha)
    elif sig == "raw":
        tau = float(cfg_curr.raw_threshold)
    elif sig == "zscore":
        # Gaussian-parametric reference: standardise by the normal scores and
        # threshold at the (1-alpha) quantile of a standard normal.
        mu, sd = n.mean(), n.std() + 1e-8
        s = (s - mu) / sd
        tau = float(_norm_ppf(1.0 - alpha))
    elif sig == "synth_percentile":
        tau = float(np.quantile(n, 1.0 - alpha))    # plain empirical quantile
    else:
        raise ValueError(f"unknown controller signal {sig}")

    if not np.isfinite(tau):
        return 1.0
    return float(np.mean(s < tau))
