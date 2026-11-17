"""Equivalence and conservatism tests — for claims that p-values cannot make.

WHY THIS MODULE EXISTS

The project's status document asserts four findings as CONFIRMED on the basis
of p > 0.05:

    Random vs Distributional (power)      delta -0.0153, p = 0.2293
    Random vs Distributional (calib)      delta -0.0005, p = 0.2524
    Conformal vs Z-score (power)          delta +0.0065, p = 0.2293
    Conformal vs Z-score (calib)          delta +0.0006, p = 0.0637

A large p-value is *absence of evidence for a difference*. It is not evidence
of absence. With 14 paired classes, p = 0.23 is equally consistent with "these
are identical" and with "we were underpowered to see a real 2-point gap". A
reviewer will say exactly that, and the whole "the architecture can be safely
simplified" contribution rests on those four rows.

The correct instrument is a TOST equivalence test: declare in advance the
largest difference you would consider scientifically negligible (the
equivalence margin, delta), then test whether the observed effect is
significantly inside +/- delta. That converts "we failed to find a difference"
into "we demonstrated the difference is smaller than delta" — which is a
positive claim, is what you actually believe, and is much harder to attack.

The same document reports the headline Phase 4 result as

    Patched ACI holds target at e=0.1 | 0.0131 | p = 4.169e-04
    "FPR is statistically indistinguishable from 0.05"

That interpretation contradicts its own p-value: p = 4e-04 against a null of
0.05 means the FPR is significantly DIFFERENT from 0.05. And 0.0131 is roughly
a quarter of the target, i.e. the threshold is far too conservative. That is
not a failure — bounding FPR below target under 10% contamination is a good
result — but it must be reported as conservatism, with the TPR paid for it,
not as on-target calibration.

USAGE

    python -m cgc_ad.equivalence results/phase5_merged.jsonl \
        --contrast ctrl_random ctrl_distributional --metric tpr_post --margin 0.03

    python -m cgc_ad.equivalence results/phase6_merged.jsonl \
        --contrast sig_conformal sig_zscore --metric tpr_post --margin 0.03

    python -m cgc_ad.equivalence results/phase4_merged.jsonl \
        --target-fpr --tag ratchet_fixed --alpha-target 0.05
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analyze import by_arm, load, value


# --------------------------------------------------------------------------
# Distributions, without a scipy dependency
# --------------------------------------------------------------------------

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _t_cdf(t: float, df: int) -> float:
    """Student-t CDF via the regularised incomplete beta function."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    ib = 0.5 * _betainc(0.5 * df, 0.5, x)
    return ib if t <= 0 else 1.0 - ib


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 200,
            eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


# --------------------------------------------------------------------------
# TOST
# --------------------------------------------------------------------------

def tost(diffs: np.ndarray, margin: float) -> Dict[str, float]:
    """Two one-sided tests for paired differences.

    H0_lower: mu <= -margin      H0_upper: mu >= +margin
    Equivalence is declared when BOTH are rejected, i.e. p_tost < 0.05, which
    is algebraically the same as the 90% CI lying entirely inside +/- margin.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return {"n": n, "p_tost": float("nan")}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 1e-12
    df = n - 1
    t_lo = (mean + margin) / se
    t_hi = (mean - margin) / se
    p_lo = 1.0 - _t_cdf(t_lo, df)          # H0: mu <= -margin
    p_hi = _t_cdf(t_hi, df)                # H0: mu >= +margin
    p = max(p_lo, p_hi)
    half = _t_quantile(0.95, df) * se      # 90% CI, the TOST-equivalent interval
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "ci90_lo": mean - half, "ci90_hi": mean + half,
            "p_lower": p_lo, "p_upper": p_hi, "p_tost": p,
            "equivalent": bool(p < 0.05),
            "margin": margin,
            "smallest_supportable_margin": _smallest_margin(mean, se, df)}


def _t_quantile(q: float, df: int, lo: float = -50.0, hi: float = 50.0) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _smallest_margin(mean: float, se: float, df: int) -> float:
    """The tightest equivalence bound the data can support at alpha=0.05.

    Report this alongside your chosen margin: it is the honest answer to
    'how small could you have shown the difference to be?'
    """
    return abs(mean) + _t_quantile(0.95, df) * se


def paired_diffs(rows: List[Dict], a: str, b: str, metric: str
                 ) -> Tuple[np.ndarray, List[str]]:
    """Seed-average within class, then pair across arms. Classes are the unit."""
    A = by_arm(rows, metric)
    if a not in A or b not in A:
        raise SystemExit(f"arm(s) missing: have {sorted(A)}")
    cls = sorted(set(A[a]) & set(A[b]))
    return np.array([A[a][c] - A[b][c] for c in cls]), cls


# --------------------------------------------------------------------------
# Conservatism / on-target testing for an FPR claim
# --------------------------------------------------------------------------

def fpr_vs_target(rows: List[Dict], tag: Optional[str], alpha_target: float,
                  margin: float, metric: str = "fpr_post") -> Dict[str, float]:
    """Is the achieved FPR on target, conservative, or blown?

    Three outcomes, and only the first two are good:
      ON TARGET     equivalent to alpha within the margin
      CONSERVATIVE  significantly BELOW alpha -- safe, but you paid TPR for it
      BLOWN         significantly ABOVE alpha -- the guarantee failed
    """
    sub = [r for r in rows if tag is None or r.get("tag") == tag]
    per = defaultdict(list)
    for r in sub:
        v = value(r, metric)
        if v is not None:
            per[r["class"]].append(v)
    x = np.array([np.mean(v) for v in per.values()])
    if x.size < 3:
        return {"n": int(x.size)}
    d = x - alpha_target
    res = tost(d, margin)
    mean, se, df = float(d.mean()), res["se"], int(res["n"] - 1)
    t = mean / se if se > 0 else 0.0
    p_two = 2 * min(_t_cdf(t, df), 1 - _t_cdf(t, df))
    if res["equivalent"]:
        verdict = "ON TARGET"
    elif p_two < 0.05 and mean < 0:
        verdict = f"CONSERVATIVE by {-mean:.4f} ({x.mean() / alpha_target:.2f}x target)"
    elif p_two < 0.05:
        verdict = f"BLOWN by {mean:.4f}"
    else:
        verdict = "INCONCLUSIVE (neither equivalent nor different)"
    res.update({"achieved_fpr": float(x.mean()), "target": alpha_target,
                "p_difference": p_two, "verdict": verdict})
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report_equivalence(name: str, res: Dict[str, float], metric: str) -> None:
    print(f"\n{name}   [{metric}]")
    if not np.isfinite(res.get("p_tost", float("nan"))):
        print(f"  too few paired units (n={res.get('n')})")
        return
    print(f"  n classes         {res['n']}")
    print(f"  mean difference   {res['mean']:+.5f}  (sd {res['sd']:.5f})")
    print(f"  90% CI            [{res['ci90_lo']:+.5f}, {res['ci90_hi']:+.5f}]")
    print(f"  equivalence margin +/- {res['margin']:.4f}")
    print(f"  TOST p            {res['p_tost']:.4g}")
    print(f"  tightest margin the data supports  +/- "
          f"{res['smallest_supportable_margin']:.5f}")
    if res["equivalent"]:
        print(f"  --> EQUIVALENT. Claim: the difference is smaller than "
              f"{res['margin']:.3f} (p={res['p_tost']:.3g}). This is a positive "
              f"result and replaces 'no significant difference'.")
    else:
        print(f"  --> NOT EQUIVALENT at this margin. You cannot claim the two "
              f"are the same; you can only say you did not detect a difference. "
              f"Either widen the margin to "
              f"{res['smallest_supportable_margin']:.3f} and justify it, or "
              f"collect more classes/seeds.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("cgc-ad-equivalence")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--contrast", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--metric", default="tpr_post")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="largest difference you would call negligible; "
                         "PRE-REGISTER this, do not tune it to get a result")
    ap.add_argument("--target-fpr", action="store_true",
                    help="test achieved FPR against the nominal alpha instead")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--alpha-target", type=float, default=0.05)
    args = ap.parse_args(argv)

    rows = load(args.paths)
    print(f"{len(rows)} rows | arms: {sorted({r.get('tag') for r in rows})}")

    if args.target_fpr:
        res = fpr_vs_target(rows, args.tag, args.alpha_target,
                            margin=0.2 * args.alpha_target)
        print(f"\nFPR vs nominal target   [tag={args.tag or 'ALL'}]")
        print(f"  achieved FPR      {res.get('achieved_fpr', float('nan')):.5f}"
              f"   target {args.alpha_target}")
        print(f"  n classes         {res.get('n')}")
        print(f"  90% CI on (FPR - target)  "
              f"[{res.get('ci90_lo', float('nan')):+.5f}, "
              f"{res.get('ci90_hi', float('nan')):+.5f}]")
        print(f"  --> {res.get('verdict')}")
        print("\n  If the verdict is CONSERVATIVE, report the TPR at this "
              "operating point in the same table. A threshold that undershoots "
              "the target buys safety with detection power, and a reviewer will "
              "ask what it cost.")
        return 0

    if not args.contrast:
        ap.error("give --contrast A B, or use --target-fpr")
    d, cls = paired_diffs(rows, args.contrast[0], args.contrast[1], args.metric)
    res = tost(d, args.margin)
    report_equivalence(f"{args.contrast[0]} vs {args.contrast[1]}", res, args.metric)
    print(f"  classes: {', '.join(cls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
