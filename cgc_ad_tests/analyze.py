"""Turn results.jsonl into the paper's tables.

    python -m cgc_ad.analyze results.jsonl                     # all sections
    python -m cgc_ad.analyze results.jsonl --contrast coupled decoupled
    python -m cgc_ad.analyze a.jsonl b.jsonl --section threshold

Statistical protocol (review F13), applied everywhere:
  * seeds are averaged WITHIN class -> one value per (arm, class);
  * classes are the unit of analysis, paired across arms;
  * Wilcoxon signed-rank across classes, Holm-Bonferroni within a section;
  * paired bootstrap CI over classes, and a win count.
Seeds are not independent replicates of a class-level effect and are never
treated as such.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

# Metrics where a LOWER value is better.
LOWER_BETTER = {"mad_fpr", "mad_fpr_all", "max_fpr_dev", "recovery_delay",
                "n_false_alarms", "n_missed", "fpr_dev"}


def load(paths: Sequence[str]) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        rows.append(r)
    return rows


def value(r: Dict, metric: str) -> Optional[float]:
    for src in (r.get("stream", {}), r.get("static_clean", {}),
                r.get("static_drift", {}), r.get("monotonicity", {}), r):
        if isinstance(src, dict) and metric in src:
            v = src[metric]
            if isinstance(v, (int, float)) and v == v and abs(v) != math.inf:
                return float(v)
            return None
    if metric == "auroc_clean":
        return r.get("static_clean", {}).get("image_auroc")
    if metric == "auroc_drift":
        return r.get("static_drift", {}).get("image_auroc")
    return None


def by_arm(rows: List[Dict], metric: str, arm_key: str = "tag"
           ) -> Dict[str, Dict[str, float]]:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = value(r, metric)
        if v is not None:
            acc[r[arm_key]][r["class"]].append(v)
    return {a: {c: float(np.mean(vs)) for c, vs in d.items()}
            for a, d in acc.items()}


def wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; EXACT for n <= 20.

    Zero differences are dropped (Wilcoxon's convention); ties share midranks,
    and the exact null is obtained by enumerating all 2^n sign assignments, so
    ties and zeros are handled correctly. The previous normal approximation
    returned 0.059 at n = 5 when the exact value is 0.0625, i.e. it was
    anti-conservative precisely where every E1/E2 contrast sat, and it
    returned NaN whenever zeros left fewer than 5 pairs.

    Floor: with n non-zero pairs, p >= 2 / 2**n. At n = 5 that is 0.0625, so
    no 5-class contrast can reach p < 0.05 whatever the effect size.
    """
    d = np.asarray([x for x in np.asarray(d, float) if np.isfinite(x) and x != 0.0])
    n = d.size
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(a):                                   # midranks for ties
        m = a == v
        ranks[m] = ranks[m].mean()
    w_obs = ranks[d > 0].sum()
    mu = ranks.sum() / 2.0
    if n <= 20:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        w = signs @ ranks
        return float(min(1.0, np.mean(np.abs(w - mu) >= abs(w_obs - mu) - 1e-12)))
    sd = math.sqrt((ranks ** 2).sum() / 4.0)
    z = (abs(w_obs - mu) - 0.5) / sd
    return float(min(1.0, 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))))


def wilcoxon_floor(n: int) -> float:
    return 2.0 / 2 ** n if n > 0 else 1.0


def holm(ps: Sequence[float]) -> List[float]:
    idx = np.argsort(ps)
    m = len(ps)
    out = [float("nan")] * m
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


def bootstrap_ci(d: np.ndarray, B: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(B, d.size), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def contrast(rows: List[Dict], a: str, b: str, metrics: Sequence[str],
             arm_key: str = "tag", title: str = "") -> None:
    print(f"\n{title or f'{a} - {b}'}")
    print(f"  {'metric':20s} {'delta':>9s} {'median':>9s} {'CI95':>21s} "
          f"{'p':>7s} {'p_holm':>7s} {'better':>8s}")
    stats, raw = [], []
    for m in metrics:
        A = by_arm(rows, m, arm_key)
        if a not in A or b not in A:
            continue
        cls = sorted(set(A[a]) & set(A[b]))
        if not cls:
            continue
        d = np.array([A[a][c] - A[b][c] for c in cls])
        better = int((d < 0).sum() if m in LOWER_BETTER else (d > 0).sum())
        stats.append((m, d, better, len(cls)))
        raw.append(wilcoxon_p(d))
    adj = holm([p if p == p else 1.0 for p in raw])
    n_cls = max((n for *_, n in stats), default=0)
    for (m, d, better, n), p, pa in zip(stats, raw, adj):
        lo, hi = bootstrap_ci(d)
        star = "*" if (pa == pa and pa < 0.05) else " "
        print(f"  {m:20s} {d.mean():+9.4f} {np.median(d):+9.4f} "
              f"[{lo:+8.4f},{hi:+8.4f}] {p:7.3f} {pa:6.3f}{star} "
              f"{better:4d}/{n:<3d}")

    if n_cls:
        print(f"  note: exact two-sided Wilcoxon floor at n={n_cls} classes is "
              f"{wilcoxon_floor(n_cls):.4g}"
              + ("  -> p < 0.05 is unattainable; report effect sizes and CIs"
                 if wilcoxon_floor(n_cls) >= 0.05 else ""))

def table(rows: List[Dict], metrics: Sequence[str], arms: Optional[List[str]] = None,
          arm_key: str = "tag") -> None:
    arms = arms or sorted({r[arm_key] for r in rows})
    print(f"\n  {'metric':20s}" + "".join(f"{a[:14]:>16s}" for a in arms))
    for m in metrics:
        A = by_arm(rows, m, arm_key)
        line = f"  {m:20s}"
        for a in arms:
            line += (f"{np.mean(list(A[a].values())):16.4f}" if a in A and A[a]
                     else f"{'-':>16s}")
        print(line)


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / d) if d > 0 else 0.0


def trend(rows: List[Dict], field: str, metric: str) -> Dict[str, float]:
    """Within-category trend test for a swept dose variable.

    WHY: pooling (category x level) cells into one Spearman correlation treats
    repeated measures on the same category as independent. With 15 categories x
    4 window lengths that gives n = 60 and a p-value ~11 orders of magnitude too
    small -- which is exactly what the reported rho = 0.794, p = 3.95e-14 implies
    (and n = 60 includes the excluded toothbrush category). The valid test
    computes one Spearman rho per category across its levels, then asks whether
    those 14 per-category correlations are centred away from zero.
    """
    from .audit import _get as get
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        lvl = get(r, f"config.{field}")
        v = value(r, metric)
        if lvl is not None and v is not None:
            per[r["class"]][lvl].append(v)
    rhos = []
    for c, d in per.items():
        levels = sorted(d)
        if len(levels) >= 3:
            rhos.append(_spearman(np.array(levels, float),
                                  np.array([np.mean(d[l]) for l in levels])))
    rhos = np.array(rhos)
    if rhos.size < 5:
        print(f"  trend {field} -> {metric}: too few categories ({rhos.size})")
        return {}
    p = wilcoxon_p(rhos)
    lo, hi = bootstrap_ci(rhos)
    print(f"  trend {field} -> {metric}: n={rhos.size} categories | "
          f"median within-category rho={np.median(rhos):+.3f} "
          f"[{lo:+.3f},{hi:+.3f}] | Wilcoxon p={p:.3g} | "
          f"{int((rhos > 0).sum())}/{rhos.size} positive")
    print(f"    (minimum attainable two-sided p at n={rhos.size} is "
          f"{2 / 2 ** rhos.size:.2e})")
    return {"n": int(rhos.size), "median_rho": float(np.median(rhos)), "p": p}


def _t_ci(x: np.ndarray):
    from .equivalence import _t_quantile
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(x.mean())
    if x.size < 2:
        return m, float("nan"), float("nan"), 1
    h = _t_quantile(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    return m, m - h, m + h, int(x.size)


def _identity(r: Dict) -> Optional[float]:
    v = value(r, "fpr_alarm_identity")
    if v is not None:
        return v
    f, t, a = value(r, "fpr_post"), value(r, "tpr_post"), value(r, "alarm_rate_post")
    alpha = (r.get("config", {}).get("threshold", {}) or {}).get("alpha", 0.05)
    if None in (f, t, a) or abs(t - f) < 1e-9:
        return None
    pi = (a - f) / (t - f)                    # alarm = (1-pi) f + pi t
    return (alpha - pi * t) / (1 - pi) if pi < 1 else None


def summary(rows: List[Dict], csv_path: Optional[str] = None) -> None:
    """Absolute per-arm operating points with class-level 95% t-intervals.

    Differences between arms hide whether ANY arm hit its target. This view
    answers the questions the paper actually rests on:
      * labelled / reviewed:  is fpr_post ~ alpha?
      * alarm-rate arms:      is alarm_rate_post ~ alpha, and is
                              fpr_post - (alpha - pi*TPR)/(1 - pi) ~ 0?
      * bursts:               does tpr_burst fall below tpr_pre_burst?
      * ceiling:              does the drifted nominal quantile exceed max B0?
    Seeds are averaged within class first; the class is the unit.
    """
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ctrl = {}
    for r in rows:
        tag = r.get("tag")
        ctrl.setdefault(tag, (r.get("stream") or {}).get("controlled_quantity", "?"))
        for m in ("fpr_post", "tpr_post", "alarm_rate_post", "prevalence_post",
                  "admission_rate_post", "tpr_pre_burst", "tpr_burst",
                  "max_B0", "nominal_q_post", "ceiling_binds", "auroc_retention",
                  "n_adaptations", "mad_fpr"):
            v = value(r, m)
            if v is not None:
                per[tag][m][r["class"]].append(float(v))
        idv = _identity(r)
        f = value(r, "fpr_post")
        if idv is not None and f is not None:
            per[tag]["identity_resid"][r["class"]].append(f - idv)
        tb, tp = value(r, "tpr_burst"), value(r, "tpr_pre_burst")
        if tb is not None and tp is not None:
            per[tag]["burst_drop"][r["class"]].append(tb - tp)

    def cell(tag, m, fmt="{:.4f}"):
        cls = per[tag].get(m, {})
        x = np.array([np.mean(v) for v in cls.values()])
        mu, lo, hi, n = _t_ci(x)
        if n == 0:
            return "      -       ", None
        return (f"{fmt.format(mu)} [{fmt.format(lo)},{fmt.format(hi)}]"
                if n > 1 else fmt.format(mu)), (mu, lo, hi, n)

    tags = sorted(per)
    ncls = {t: len(per[t].get("fpr_post", {})) for t in tags}
    print("\nOPERATING POINT (class-level mean [95% t-CI])")
    print(f"  {'arm':28s} {'n':>2s}  {'FPR':^26s} {'TPR':^26s} {'alarm rate':^26s}  controls")
    for t in tags:
        print(f"  {t:28s} {ncls[t]:2d}  {cell(t,'fpr_post')[0]:26s} "
              f"{cell(t,'tpr_post')[0]:26s} {cell(t,'alarm_rate_post')[0]:26s}  "
              f"{str(ctrl[t])[:34]}")
    print("\nALARM-RATE IDENTITY  FPR - (alpha - pi*TPR)/(1 - pi)   "
          "[~0 for arms fed all-sample alarms]")
    for t in tags:
        c, _ = cell(t, "identity_resid", "{:+.4f}")
        pi, _ = cell(t, "prevalence_post")
        print(f"  {t:28s} resid {c:32s} realised pi {pi}")
    if any(per[t].get("burst_drop") for t in tags):
        print("\nDEFECT BURST  (TPR during burst - TPR before burst; negative = desensitised)")
        for t in tags:
            if per[t].get("burst_drop"):
                print(f"  {t:28s} before {cell(t,'tpr_pre_burst')[0]:26s} "
                      f"during {cell(t,'tpr_burst')[0]:26s} "
                      f"drop {cell(t,'burst_drop','{:+.3f}')[0]}")
    if any(per[t].get("ceiling_binds") for t in tags):
        print("\nCEILING (Prop. 2): does the drifted nominal (1-alpha) quantile exceed max B0?")
        for t in tags:
            if per[t].get("ceiling_binds"):
                print(f"  {t:28s} binds in {cell(t,'ceiling_binds','{:.2f}')[0]:22s} "
                      f"max_B0 {cell(t,'max_B0','{:.3f}')[0]:24s} "
                      f"nominal q {cell(t,'nominal_q_post','{:.3f}')[0]:24s} "
                      f"retention {cell(t,'auroc_retention','{:+.3f}')[0]}")
    print("\nCONFOUND CHECK: realised adaptation events per run (should be 0 "
          "for threshold studies)")
    for t in tags:
        print(f"  {t:28s} {cell(t,'n_adaptations','{:.2f}')[0]}")

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "metric", "n_classes", "mean", "ci_lo", "ci_hi"])
            for t in tags:
                for m in per[t]:
                    _, st = cell(t, m)
                    if st:
                        w.writerow([t, m, st[3], *[f"{v:.6g}" for v in st[:3]]])
        print(f"\nwrote {csv_path}")


def gates(rows: List[Dict]) -> None:
    print("\nGATES")
    mono = [(r["class"], r["monotonicity"]["spearman"], r["monotonicity"]["range"])
            for r in rows if "monotonicity" in r]
    ok = [1 for _, s, g in mono if s > 0.8 and g > 0.3]
    print(f"  monotonicity: {len(ok)}/{len(mono)} runs pass "
          f"(spearman>0.8 and range>0.3)")
    per = defaultdict(list)
    for c, s, g in mono:
        per[c].append(s > 0.8 and g > 0.3)
    bad = [(c, float(np.mean(v))) for c, v in per.items() if np.mean(v) < 0.8]
    if bad:
        print("    FAILING classes (hardness axis is ill-posed here): "
              + ", ".join(f"{c}={p:.2f}" for c, p in sorted(bad, key=lambda x: x[1])))

    A = by_arm(rows, "auroc_clean")
    for a in sorted(A):
        mu = np.mean(list(A[a].values()))
        flag = "" if mu >= 0.95 else "   <-- BELOW PARITY (CutPaste/PatchCore ~0.95-0.99)"
        print(f"  clean image AUROC [{a:16s}] = {mu:.4f}{flag}")

    seen = {}
    for r in rows:
        seen.setdefault(r["class"], r)
    small = [(c, r["data"]["n_calib"], 1.0 / (r["data"]["n_calib"] + 1))
             for c, r in seen.items()]
    bad = [(c, n, a) for c, n, a in small if a > 0.05]
    if bad:
        print("  UNATTAINABLE alpha=0.05 (n_calib too small, review F6): "
              + ", ".join(f"{c} n={n} alpha_min={a:.3f}" for c, n, a in bad))


def admit_check(rows: List[Dict]) -> None:
    """Flag runs whose buffer starved -- their stream metrics are meaningless."""
    bad = []
    for r in rows:
        s = r.get("stream", {})
        cfg = r.get("config", {}).get("stream", {})
        if s.get("fpr_post", 0) > 0.5 and cfg.get("buffer_protocol") != "clean":
            bad.append((r["tag"], r["class"], s["fpr_post"]))
    if bad:
        print(f"\n  WARNING: {len(bad)} runs have post-drift FPR > 0.5 with a "
              f"non-clean buffer. Check the admission rate before reading any "
              f"other metric from them.")
        for t, c, f in bad[:8]:
            print(f"    {t:28s} {c:12s} fpr_post={f:.3f}")


SECTIONS = {
    "gates": None,
    "rq3": (["coupled", "decoupled", "ctrl_random"],
            ["tpr_post", "stream_auroc_post", "mad_fpr", "fpr_post",
             "auroc_clean", "auroc_drift", "recovery_delay"]),
    "threshold": (["thr_static", "thr_window", "thr_aci", "thr_robust_aci"],
                  ["mad_fpr", "max_fpr_dev", "fpr_post", "tpr_post",
                   "recovery_delay", "n_false_alarms", "n_missed"]),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("cgc-ad-analyze")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--section", default="all",
                    choices=["all", "gates", "rq3", "threshold"])
    ap.add_argument("--contrast", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--summary", action="store_true",
                    help="absolute per-arm operating points, identity check, "
                         "burst and ceiling tables")
    ap.add_argument("--csv", default=None, help="with --summary: write a CSV")
    ap.add_argument("--trend", nargs=2, default=None, metavar=("FIELD", "METRIC"),
                    help="within-category trend test, e.g. threshold.window mad_fpr")
    args = ap.parse_args(argv)

    rows = load(args.paths)
    print(f"{len(rows)} rows | arms: {sorted({r['tag'] for r in rows})} | "
          f"classes: {len(set(r['class'] for r in rows))} | "
          f"seeds: {sorted({r['seed'] for r in rows})}")

    if args.summary:
        summary(rows, args.csv)
        return 0

    if args.trend:
        trend(rows, args.trend[0], args.trend[1])
        return 0

    if args.contrast:
        contrast(rows, args.contrast[0], args.contrast[1],
                 args.metrics or SECTIONS["rq3"][1])
        return 0

    if args.section in ("all", "gates"):
        gates(rows)
        admit_check(rows)

    if args.section in ("all", "rq3"):
        arms, mets = SECTIONS["rq3"]
        present = [a for a in arms if any(r["tag"] == a for r in rows)]
        if len(present) >= 2:
            print("\n" + "=" * 88 + "\nRQ3: COUPLING")
            table(rows, mets, present)
            if "coupled" in present and "decoupled" in present:
                contrast(rows, "coupled", "decoupled", mets,
                         title="coupled - decoupled  [isolates adaptation-time coupling]")
            if "ctrl_random" in present and "coupled" in present:
                contrast(rows, "coupled", "ctrl_random", mets,
                         title="coupled - random     [isolates the TRAINING controller]")
            if "ctrl_random" in present and "decoupled" in present:
                contrast(rows, "ctrl_random", "decoupled", mets,
                         title="random - decoupled   [CONFOUNDED: differs in two ways]")

    if args.section in ("all", "threshold"):
        arms, mets = SECTIONS["threshold"]
        present = [a for a in arms if any(r["tag"] == a for r in rows)]
        if len(present) >= 2:
            print("\n" + "=" * 88 + "\nTHRESHOLD POLICIES")
            table(rows, mets, present)
            if "thr_aci" in present and "thr_window" in present:
                contrast(rows, "thr_aci", "thr_window", ["mad_fpr", "tpr_post"],
                         title="aci - window   [the power/calibration trade-off]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
