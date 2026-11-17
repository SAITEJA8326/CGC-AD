"""Figures. Each one exists to make a specific claim falsifiable on sight.

    python -m cgc_ad.figures phase*.jsonl --out figs/
    python -m cgc_ad.figures phase3_threshold.jsonl --figs trajectory,tradeoff \
        --traces /kaggle/working/cgc_ad_runs/runs

Figure -> claim it tests:
  f1_trajectory   tau and windowed FPR around drift onset, one line per policy.
                  THE money figure. Shows whether a threshold tracks or runs
                  away, and makes buffer starvation visible instantly.
  f2_tradeoff     MAD-FPR vs TPR, one point per policy. The power/calibration
                  trade-off that AUROC is mathematically unable to show.
  f3_breakdown    metric vs contamination, one line per buffer protocol.
                  Where the breakdown point is, and whether mitigations move it.
  f4_monotonicity miss-rate vs hardness per class. The Phase-1 gate. Classes
                  with a flat or non-monotone curve cannot support a setpoint
                  controller and must be reported as such.
  f5_paired       per-class slope plot for a two-arm contrast. Shows whether a
                  mean difference is a consistent effect or two outliers.
  f6_retention    AUROC retention vs drift severity per class. Locates the
                  regime where the detector degrades rather than collapses.
  f7_gates        clean AUROC vs the parity line, and n_calib vs attainable
                  alpha. One glance at whether the study is interpretable.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

LOWER_BETTER = {"mad_fpr", "max_fpr_dev", "recovery_delay", "n_false_alarms",
                "n_missed"}


def load(paths: Sequence[str]) -> List[Dict]:
    rows = []
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            with open(q) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if "error" not in r:
                            rows.append(r)
    return rows


def _get(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _by_class(rows, tag, metric):
    acc = defaultdict(list)
    for r in rows:
        if r.get("tag") != tag:
            continue
        v = _get(r, f"stream.{metric}", _get(r, f"static_clean.{metric}",
                                             r.get(metric)))
        if isinstance(v, (int, float)) and v == v and abs(v) != math.inf:
            acc[r["class"]].append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def _rolling(x, w):
    if x.size == 0:
        return x
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    i = np.arange(x.size)
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i - lo + 1)


# ==========================================================================
# f1 -- threshold trajectory  (the money figure)
# ==========================================================================

def f1_trajectory(rows, out, traces_root=None, cls=None, window=100, alpha=0.05):
    """tau and windowed FPR through the drift onset, one line per arm.

    Reads trace.npz written by the runner. If a run's admission rate collapses,
    it is visible here as a flat tau plus an FPR pinned near 1 -- which is how
    the self-labelling ratchet was identified.
    """
    if not traces_root:
        return None
    paths = {}
    for r in rows:
        c = r.get("class")
        if cls and c != cls:
            continue
        d = r.get("run_dir") or ""
        p = os.path.join(d, "trace.npz")
        if os.path.exists(p):
            paths.setdefault(r.get("tag"), p)
    if not paths:
        hits = sorted(glob.glob(os.path.join(traces_root, "*", "*",
                                             cls or "*", "*", "trace.npz")))
        for p in hits:
            paths.setdefault(p.split(os.sep)[-5], p)
    if not paths:
        print("  f1: no trace.npz found; skipping")
        return None

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7.5, 7))
    for tag, p in sorted(paths.items()):
        d = np.load(p)
        t = d["t"]
        tau = np.where(np.isfinite(d["tau"]), d["tau"], np.nan)
        ax[0].plot(t, tau, lw=1.1, label=tag)
        nrm = d["label"] == 0
        ax[1].plot(t[nrm], _rolling(d["flag"][nrm].astype(float), window), lw=1.1)
        ax[2].plot(t, _rolling(d["admitted"].astype(float), window), lw=1.1)
        onset = int(t[np.argmax(d["severity"] > 0)]) if (d["severity"] > 0).any() \
            else None

    for a in ax:
        if onset:
            a.axvline(onset, color="k", ls=":", lw=1)
    ax[1].axhline(alpha, color="k", ls="--", lw=1)
    ax[1].axhspan(0.5 * alpha, 1.5 * alpha, color="k", alpha=0.06)
    ax[2].axhline(0.25, color="crimson", ls="--", lw=1)
    ax[0].set_ylabel(r"threshold $\tau_t$")
    ax[1].set_ylabel(f"windowed FPR (w={window})")
    ax[2].set_ylabel("buffer admission rate")
    ax[2].set_xlabel("stream step")
    ax[2].text(0.01, 0.08, "below the dashed line the buffer is starving:\n"
                           "every metric from that run is invalid",
               transform=ax[2].transAxes, color="crimson", fontsize=7.5)
    ax[0].legend(fontsize=7.5, ncol=2)
    ax[0].set_title(f"Threshold trajectory through drift onset"
                    + (f" -- {cls}" if cls else ""))
    return _save(fig, out, "f1_trajectory")


# ==========================================================================
# f2 -- power / calibration trade-off
# ==========================================================================

def f2_tradeoff(rows, out, x="mad_fpr", y="tpr_post"):
    """One point per arm: calibration error against detection power.

    This is the figure AUROC cannot produce. An arm that holds nominal FPR by
    raising tau until nothing fires appears here in the bottom-left and is
    immediately recognisable as a bad trade.
    """
    tags = sorted({r.get("tag") for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for tag in tags:
        X, Y = _by_class(rows, tag, x), _by_class(rows, tag, y)
        cs = sorted(set(X) & set(Y))
        if not cs:
            continue
        mx, my = np.mean([X[c] for c in cs]), np.mean([Y[c] for c in cs])
        sx = np.std([X[c] for c in cs]) / max(1, np.sqrt(len(cs)))
        sy = np.std([Y[c] for c in cs]) / max(1, np.sqrt(len(cs)))
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="o", ms=7, capsize=3)
        ax.annotate(tag, (mx, my), textcoords="offset points", xytext=(7, 4),
                    fontsize=8)
    ax.set_xlabel(f"{x}  (lower = better calibrated)")
    ax.set_ylabel(f"{y}  (higher = more detection power)")
    ax.set_title("Power vs calibration\n(mean over classes, SE bars)")
    ax.annotate("better", xy=(0.06, 0.94), xycoords="axes fraction",
                fontsize=8, color="green")
    return _save(fig, out, "f2_tradeoff")


# ==========================================================================
# f3 -- contamination breakdown
# ==========================================================================

def f3_breakdown(rows, out, sweep_field="stream.contamination",
                 metrics=("fpr_post", "tpr_post", "mad_fpr")):
    """Metric vs the sweep variable, one line per buffer protocol.

    Before plotting, this checks that `sweep_field` actually varies. If it does
    not, it refuses to draw -- the first attempt at this figure swept
    `prevalence` while `contamination` stayed at 0.00, and a plot would have
    made that invisible.
    """
    pts = defaultdict(lambda: defaultdict(list))
    xs = set()
    for r in rows:
        cfg = r.get("config", {})
        x = _get(cfg, sweep_field)
        proto = _get(cfg, "stream.buffer_protocol", "?")
        upd = _get(cfg, "threshold.aci_update_on", "admitted")
        expl = _get(cfg, "stream.admit_explore", 0.0) or 0.0
        if not isinstance(x, (int, float)):
            continue
        xs.add(float(x))
        label = f"{proto}" + ("" if proto == "clean" else
                              f" [{upd},expl={expl:g}]")
        for m in metrics:
            v = _get(r, f"stream.{m}")
            if isinstance(v, (int, float)) and v == v:
                pts[label][(m, float(x))].append(v)
    if len(xs) < 2:
        print(f"  f3: `{sweep_field}` takes only {len(xs)} value(s) across "
              f"these rows -- nothing was swept. Refusing to plot.")
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    for j, m in enumerate(metrics):
        for label, d in sorted(pts.items()):
            k = sorted([x for (mm, x) in d if mm == m])
            if not k:
                continue
            y = [np.mean(d[(m, x)]) for x in k]
            e = [np.std(d[(m, x)]) / max(1, np.sqrt(len(d[(m, x)]))) for x in k]
            axes[j].errorbar(k, y, yerr=e, marker="o", ms=4, capsize=2,
                             lw=1.2, label=label)
        axes[j].set_xlabel(sweep_field.split(".")[-1])
        axes[j].set_ylabel(m)
        if m == "fpr_post":
            axes[j].axhline(0.05, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=7)
    fig.suptitle("Buffer-protocol breakdown", y=1.02)
    return _save(fig, out, "f3_breakdown")


# ==========================================================================
# f4 -- monotonicity gate
# ==========================================================================

def f4_monotonicity(rows, out):
    """Miss-rate vs hardness per class, failures highlighted.

    A flat or non-monotone curve means the setpoint controller is chasing an
    ill-posed target on that class. In the first round 5 of 15 classes failed,
    and `toothbrush` had a range of exactly 0.000.
    """
    per = defaultdict(list)
    lv = None
    for r in rows:
        m = r.get("monotonicity")
        if m and m.get("miss_rate"):
            per[r["class"]].append(m["miss_rate"])
            lv = m["levels"]
    if not per:
        return None
    n = len(per)
    ncol = min(5, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.1 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for a, (cls, curves) in zip(axes, sorted(per.items())):
        y = np.mean(curves, axis=0)
        rng = float(y.max() - y.min())
        mono = bool(np.all(np.diff(y) >= -1e-3))
        ok = mono and rng > 0.3
        a.plot(lv, y, marker="o", ms=3, color="tab:blue" if ok else "crimson")
        a.set_title(f"{cls}\n{'PASS' if ok else 'FAIL'} range={rng:.2f}",
                    fontsize=8, color="black" if ok else "crimson")
    for a in axes[n:]:
        a.axis("off")
    fig.supxlabel("hardness h", fontsize=9)
    fig.supylabel("miss-rate at the conformal threshold", fontsize=9)
    fig.suptitle("Phase-1 gate: is hardness monotone in detectability?", y=1.01)
    return _save(fig, out, "f4_monotonicity")


# ==========================================================================
# f5 -- paired per-class contrast
# ==========================================================================

def f5_paired(rows, out, a="coupled", b="decoupled", metric="tpr_post"):
    """Slope plot: one line per class, seed-averaged.

    A mean difference driven by two classes looks identical to a consistent
    effect in a bar chart. It does not look identical here.
    """
    A, B = _by_class(rows, a, metric), _by_class(rows, b, metric)
    cs = sorted(set(A) & set(B))
    if len(cs) < 3:
        return None
    better_is_low = metric in LOWER_BETTER
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    wins = 0
    for c in cs:
        d = A[c] - B[c]
        good = (d < 0) if better_is_low else (d > 0)
        wins += bool(good)
        ax.plot([0, 1], [B[c], A[c]], marker="o", ms=4, lw=1.1,
                color="tab:blue" if good else "crimson", alpha=0.8)
        ax.annotate(c, (1.02, A[c]), fontsize=6.5, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels([b, a])
    ax.set_ylabel(metric)
    d = np.array([A[c] - B[c] for c in cs])
    ax.set_title(f"{a} vs {b}: {metric}\n"
                 f"mean {d.mean():+.4f}, better on {wins}/{len(cs)} classes",
                 fontsize=9)
    ax.set_xlim(-0.15, 1.35)
    return _save(fig, out, f"f5_paired_{a}_vs_{b}_{metric}")


# ==========================================================================
# f6 -- drift severity vs retention
# ==========================================================================

def f6_retention(rows, out):
    """AUROC retention against drift severity, per class.

    Identifies the usable regime. Retention around -0.27 (the first round) means
    the detector is being destroyed, not drifted, and threshold comparisons made
    there say very little.
    """
    pts = defaultdict(list)
    for r in rows:
        s = _get(r, "config.drift.severity")
        v = r.get("auroc_retention")
        if isinstance(s, (int, float)) and isinstance(v, (int, float)):
            pts[r["class"]].append((float(s), float(v)))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    allsev = sorted({s for v in pts.values() for s, _ in v})
    for cls, v in sorted(pts.items()):
        d = defaultdict(list)
        for s, y in v:
            d[s].append(y)
        k = sorted(d)
        ax.plot(k, [np.mean(d[s]) for s in k], marker="o", ms=3, lw=1,
                alpha=0.75, label=cls)
    ax.axhline(-0.15, color="crimson", ls="--", lw=1)
    ax.text(0.02, 0.06, "below this line the detector is collapsing,\nnot drifting",
            transform=ax.transAxes, color="crimson", fontsize=7.5)
    ax.set_xlabel("drift severity")
    ax.set_ylabel("AUROC retention  (drift - clean)")
    ax.set_title("Choosing the usable drift regime")
    if len(pts) <= 16:
        ax.legend(fontsize=6, ncol=3)
    if len(allsev) < 2:
        ax.text(0.5, 0.5, "only one severity present:\nsweep drift.severity",
                transform=ax.transAxes, ha="center", color="crimson")
    return _save(fig, out, "f6_retention")


# ==========================================================================
# f7 -- gate dashboard
# ==========================================================================

def f7_gates(rows, out, parity=0.95):
    """Clean AUROC against the parity line, and calibration size against the
    attainable alpha. Both gates on one page."""
    per = defaultdict(list)
    ncal, alpha = {}, None
    for r in rows:
        v = _get(r, "static_clean.image_auroc")
        if isinstance(v, (int, float)):
            per[r["class"]].append(v)
        if r.get("class") not in ncal:
            n = _get(r, "data.n_calib")
            if n:
                ncal[r["class"]] = n
        alpha = alpha or _get(r, "config.threshold.alpha", 0.05)
    if not per:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    cs = sorted(per, key=lambda c: np.mean(per[c]))
    y = [np.mean(per[c]) for c in cs]
    ax[0].barh(cs, y, color=["crimson" if v < parity else "tab:blue" for v in y])
    ax[0].axvline(parity, color="k", ls="--", lw=1)
    ax[0].set_xlim(0.5, 1.0)
    ax[0].set_xlabel("clean image AUROC")
    ax[0].set_title(f"Parity gate (CutPaste ~{parity:.2f}, PatchCore ~0.99)")

    if ncal:
        cs2 = sorted(ncal, key=lambda c: ncal[c])
        amin = [1.0 / (ncal[c] + 1) for c in cs2]
        ax[1].barh(cs2, amin,
                   color=["crimson" if a > alpha else "tab:blue" for a in amin])
        ax[1].axvline(alpha, color="k", ls="--", lw=1)
        ax[1].set_xlabel(r"smallest attainable $\alpha$ = 1/(n_calib+1)")
        ax[1].set_title(f"Attainability gate (target alpha={alpha})\n"
                        f"red bars: this class cannot express the target rate")
    return _save(fig, out, "f7_gates")


# ==========================================================================

def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"{name}.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


FIGS = {"trajectory": f1_trajectory, "tradeoff": f2_tradeoff,
        "breakdown": f3_breakdown, "monotonicity": f4_monotonicity,
        "paired": f5_paired, "retention": f6_retention, "gates": f7_gates}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("cgc-ad-figures")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--figs", default="all")
    ap.add_argument("--traces", default=None,
                    help="root of runs/ for trace.npz lookup (figure f1)")
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--contrast", nargs=2, default=["coupled", "decoupled"])
    ap.add_argument("--metric", default="tpr_post")
    ap.add_argument("--sweep-field", default="stream.contamination")
    args = ap.parse_args(argv)

    rows = load(args.paths)
    print(f"{len(rows)} rows | tags={sorted({r.get('tag') for r in rows})}")
    want = list(FIGS) if args.figs == "all" else args.figs.split(",")

    for name in want:
        fn = FIGS.get(name)
        if fn is None:
            print(f"  unknown figure `{name}`")
            continue
        try:
            if name == "trajectory":
                fn(rows, args.out, traces_root=args.traces, cls=args.cls)
            elif name == "paired":
                fn(rows, args.out, args.contrast[0], args.contrast[1], args.metric)
            elif name == "breakdown":
                fn(rows, args.out, sweep_field=args.sweep_field)
            else:
                fn(rows, args.out)
        except Exception as e:
            print(f"  {name}: skipped ({e!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
