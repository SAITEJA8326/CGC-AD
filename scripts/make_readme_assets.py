#!/usr/bin/env python3
"""Regenerate every figure used in README.md.

    python scripts/make_readme_assets.py                 # from the repo root
    python scripts/make_readme_assets.py --results tests/results --out assets

* results_*.png  are computed from the JSONL files in tests/results (R3 sweeps:
  e1_e3_results.jsonl and e2_e4_results.jsonl), exactly as in the manuscript.
* ratchet_sim.png is an *illustrative* numpy simulation that drives the real
  threshold classes in cgc_ad/conformal.py (no GPU, ~1 minute).
* banner / architecture / censoring are drawn with matplotlib only.

Needs: numpy, matplotlib.  SciPy is NOT required.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

NAVY, MAROON, GREEN, GOLD = "#1F3A5F", "#9B1C2B", "#2F6B3A", "#C99A2E"
INK, MUTED, GRID = "#1B2433", "#6B7686", "#E4E8EE"
COL = {"LAB": "#4C6A92", "REV": "#2F8F5B", "ALR": "#E08A1E", "SL-X": "#8E5BB5",
       "SL": "#B08BD0", "NAIVE": "#C8392B", "STATIC": "#8A93A0"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#9AA3AF",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ----------------------------------------------------------------- stats helpers
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}


def t_ci(x, table):
    x = np.asarray(x, float)
    n = len(x)
    return table[n - 1] * x.std(ddof=1) / math.sqrt(n)


def load_jsonl(path):
    """Tolerant JSONL reader: a few files contain two objects on one line."""
    dec, rows = json.JSONDecoder(), []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        while line:
            obj, i = dec.raw_decode(line)
            rows.append(obj)
            line = line[i:].lstrip()
    return rows


def class_means(rows, tag, metric):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("tag") == tag and "stream" in r:
            by[r["class"]].append(r["stream"][metric])
    return {c: float(np.nanmean(v)) for c, v in by.items()}


def paired(rows, a, b, metric):
    da, db = class_means(rows, a, metric), class_means(rows, b, metric)
    cls = sorted(set(da) & set(db))
    return np.array([da[c] - db[c] for c in cls])


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=NAVY, fs=9.5, bold=False, color=INK, lw=1.4, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.25)


def arrow(ax, p, q, color=NAVY, lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))


# ----------------------------------------------------------------------- banner
def make_banner(out):
    fig = plt.figure(figsize=(16, 4.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 16, 4.2, color=NAVY, zorder=0))
    # soft score-distribution motif on the right
    x = np.linspace(9.2, 15.6, 400)
    z = (x - 11.6) / 0.95
    y = 0.55 + 2.35 * np.exp(-z ** 2 / 2)
    thr = 12.95
    ax.fill_between(x, 0.55, y, where=x <= thr, color="#5B7BA8", alpha=0.55, lw=0, zorder=1)
    ax.fill_between(x, 0.55, y, where=x >= thr, color="#E4572E", alpha=0.85, lw=0, zorder=1)
    ax.plot(x, y, color="white", lw=1.6, alpha=0.9, zorder=2)
    ax.plot([thr, thr], [0.55, 3.35], color=GOLD, lw=2.2, ls=(0, (5, 3)), zorder=3)
    ax.text(thr + 0.08, 3.42, r"threshold $\tau$", color=GOLD, fontsize=12, va="bottom")
    ax.annotate("the missing tail:\nnormal false alarms", xy=(13.7, 0.95), xytext=(14.1, 2.35), color="white",
                fontsize=11, ha="left", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2))
    ax.plot([9.0, 15.8], [0.55, 0.55], color="white", lw=1.2, alpha=0.8)
    ax.text(0.7, 2.95, "CGC-AD", color="white", fontsize=54, fontweight="bold", va="center", zorder=3)
    ax.text(0.75, 1.95, "Label-free conformal thresholds for streaming\nvisual anomaly detection",
            color="#DCE6F5", fontsize=18, va="center", linespacing=1.35, zorder=3)
    ax.text(0.75, 0.98, "What can an adaptive threshold actually control\nwhen nobody tells it which parts were normal?",
            color=GOLD, fontsize=12.5, va="center", zorder=3, style="italic", linespacing=1.4)
    ax.text(0.75, 0.33, "adaptive conformal inference  ·  streaming anomaly detection  ·  MVTec AD",
            color="#AFC0D8", fontsize=10, va="center", zorder=3)
    fig.savefig(out, dpi=110, bbox_inches=None, facecolor=NAVY)
    plt.close(fig)


# -------------------------------------------------------------------- censoring
def make_censoring(out):
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    x = np.linspace(-3.4, 6.0, 800)
    nom = np.exp(-x ** 2 / 2)
    dfc = 0.28 * np.exp(-(x - 3.4) ** 2 / (2 * 0.9 ** 2))
    thr = 1.645
    ax.fill_between(x, 0, nom, where=x <= thr, color="#5B7BA8", alpha=0.35, label="what the buffer sees (admitted: score ≤ τ)")
    ax.fill_between(x, 0, nom, where=x >= thr, color="#E4572E", alpha=0.75, label="the missing tail = nominal false alarms")
    ax.plot(x, nom, color=NAVY, lw=2)
    ax.plot(x, dfc, color=MAROON, lw=2, ls="--", label="defect scores")
    ax.axvline(thr, color=GOLD, lw=2.4, ls=(0, (5, 3)))
    ax.text(thr - 0.08, 1.02, "threshold τ", color="#8A6A12", fontsize=11, fontweight="bold", ha="right")
    ax.annotate("a threshold that decides its own\ncalibration data never sees this part", xy=(2.45, 0.07),
                xytext=(3.2, 0.62), fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.set_xlabel("anomaly score"); ax.set_ylabel("density"); ax.set_yticks([]); ax.set_ylim(0, 1.18)
    ax.set_title("Why every label-free shortcut fails: the buffer is censored", fontsize=13, loc="left",
                 fontweight="bold", color=NAVY)
    ax.grid(False)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(out); plt.close(fig)


# ----------------------------------------------------------------- architecture
def make_architecture(out):
    fig = plt.figure(figsize=(13, 8.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 8.2); ax.axis("off")

    def band(y, h, title, color, right=False):
        ax.add_patch(FancyBboxPatch((0.2, y), 12.6, h, boxstyle="round,pad=0,rounding_size=0.14",
                                    fc=color, ec="none", alpha=0.55, zorder=0))
        ax.text(12.55 if right else 0.45, y + h - 0.2, title, fontsize=11.5, fontweight="bold", color=NAVY, va="center", ha="right" if right else "left", zorder=1)

    # --- offline
    band(5.75, 2.2, "1 · OFFLINE  —  once per category and seed", "#E8EEF7")
    xs = [0.45, 3.6, 6.75, 9.9]
    labels = ["MVTec AD images\ntrain/good · test/<defect>", "Split roles\nnormal · calib · val\nsynthetic anomalies",
              "Frozen backbone\nWideResNet-50 (layer2 + layer3)\nfeature banks cached", "PatchHead  (1×1 convs, 33,153 params)\ncurriculum on synthetic hardness\n8,000 steps → head cache"]
    for x, t in zip(xs, labels):
        box(ax, x, 5.9, 2.85 if x < 9 else 2.85, 1.25, t, fc="white", fs=9)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 2.85, 6.52), (b, 6.52))
    # --- online
    band(2.55, 3.0, "2 · ONLINE  —  1,500-step stream per arm  (drift, defect bursts)", "#EDF5EE")
    box(ax, 0.45, 3.9, 2.2, 1.0, "Stream generator\nπ = 5% defects\ndrift · bursts", fc="white", ec=GREEN, fs=9)
    box(ax, 3.1, 3.9, 2.0, 1.0, "PatchHead score s_t\n(from the head cache)", fc="white", ec=GREEN, fs=8.8)
    box(ax, 5.55, 3.9, 2.0, 1.0, "Flag?\ns_t > τ", fc="#FFF6DD", ec=GOLD, fs=10, bold=True)
    box(ax, 8.0, 3.9, 2.35, 1.0, "Admission rule\nclean · reviewed\nself_label (+ explore ρ)", fc="white", ec=MAROON, fs=8.8, bold=False)
    box(ax, 10.8, 3.9, 1.85, 1.0, "FIFO buffer\n(capacity W = 200)", fc="white", ec=GREEN, fs=9)
    box(ax, 8.0, 2.7, 4.65, 0.85, "Threshold policy:  τ_t = quantile of the buffer at level 1 − α_t\nstatic  |  window  |  ACI  |  robust ACI", fc="#DFF0E3", ec=GREEN, fs=8.8, bold=True)
    box(ax, 3.1, 2.7, 3.9, 0.85, "Conformal test martingale\n→ head adaptation (100 steps)", fc="white", ec=GREEN, fs=8.8)
    for a, b in [(2.65, 3.1), (5.1, 5.55), (7.55, 8.0), (10.35, 10.8)]:
        arrow(ax, (a, 4.4), (b, 4.4), color=GREEN)
    arrow(ax, (11.7, 3.9), (11.7, 3.53), color=GREEN)
    arrow(ax, (8.0, 3.1), (6.55, 3.9), color=GREEN, rad=0.0)
    arrow(ax, (4.1, 3.9), (4.8, 3.53), color=GREEN, ls="--")
    ax.text(7.2, 3.55, "τ", color=GREEN, fontsize=11, fontweight="bold")
    ax.text(8.0, 5.05, "REV only: a person reviews flagged parts", fontsize=8.6, color=MAROON, style="italic")
        # --- evaluation
    band(0.2, 2.1, "3 · EVALUATION  —  every run is logged, audited and compared", "#FBF3E4", right=True)
    ev = ["results.jsonl\none row per run", "audit (gates A1–A7)\nintegrity · degeneracy · starvation",
          "analyze\nclass means · bootstrap CI · Holm", "equivalence\nTOST (±0.01 FPR)", "figures\nf1 … f7"]
    ex = [0.45, 2.95, 5.45, 7.95, 10.45]
    for x, t in zip(ex, ev):
        box(ax, x, 0.35, 2.25 if x < 10 else 2.2, 1.15, t, fc="white", ec=GOLD, fs=8.8)
    for a, b in zip(ex[:-1], ex[1:]):
        arrow(ax, (a + 2.25, 0.92), (b, 0.92), color="#9A7414")
    arrow(ax, (1.55, 3.85), (1.55, 1.52), color="#9A7414")
    ax.text(1.68, 2.35, "run records", fontsize=8.8, color="#9A7414", style="italic")
    fig.savefig(out); plt.close(fig)


# ----------------------------------------------------- illustrative ratchet sim
def run_arm(arm, seed, T=6000, pi=0.05, mu=2.4, W=200, alpha=0.05, gamma=0.01, explore=0.05):
    from cgc_ad.config import ThresholdConfig
    from cgc_ad.conformal import build_threshold
    rng = np.random.default_rng(seed)
    cal = rng.normal(0, 1, W)
    pol = "static" if arm == "STATIC" else "aci"
    upd = "admitted" if arm in ("LAB", "REV", "NAIVE") else "all"
    thr = build_threshold(ThresholdConfig(policy=pol, alpha=alpha, window=W, gamma_aci=gamma,
                                          aci_update_on=upd), cal)
    delta = np.clip((np.arange(T) - 2000) / 1000.0, 0, 1) * 3.0
    lab = (rng.random(T) < pi).astype(int)
    s = rng.normal(0, 1, T) + delta + lab * mu
    explore_u = rng.random(T)
    tau, fl = np.zeros(T), np.zeros(T)
    for t in range(T):
        f = thr.flag(float(s[t])); tau[t] = thr.tau; fl[t] = f
        if arm in ("LAB", "ALR"):
            adm = lab[t] == 0
        elif arm == "REV":
            adm = (not f) or lab[t] == 0
        elif arm == "STATIC":
            adm = False
        else:
            adm = not f
            if not adm and arm == "SL-X" and explore_u[t] < explore:
                adm = True
        thr.observe(float(s[t]), bool(adm), flagged=bool(f))
    return tau, fl, lab, delta, float(np.max(cal))


def make_ratchet(out, seeds=10):
    arms = ["STATIC", "NAIVE", "SL", "SL-X", "REV", "LAB"]
    names = {"STATIC": "STATIC (never adapts)", "NAIVE": "NAIVE self-labelling", "SL": "SL (all-sample ACI)",
             "SL-X": "SL-X (+5% exploration)", "REV": "REV (reviewed alarms)", "LAB": "LAB (oracle labels)"}
    T, win = 6000, 400
    taus, fprs = {}, {}
    for a in arms:
        tl, fl_all = [], []
        for sd in range(seeds):
            tau, fl, lab, delta, cmax = run_arm(a, sd)
            tl.append(tau)
            nrm = (lab == 0)
            num = np.convolve(fl * nrm, np.ones(win), "same"); den = np.convolve(nrm.astype(float), np.ones(win), "same")
            fl_all.append(num / np.maximum(den, 1))
        taus[a] = np.mean(tl, 0); fprs[a] = np.mean(fl_all, 0)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    t = np.arange(T)
    ax = axs[0]
    ax.axvspan(2000, 3000, color="#EEF1F6", zorder=0); ax.text(2500, 5.6, "drift ramp\n(+3σ)", ha="center", fontsize=9.5, color=MUTED, va="top")
    q95 = 1.645 + np.clip((t - 2000) / 1000.0, 0, 1) * 3.0
    ax.plot(t, q95, color="black", lw=1.2, ls=":", label="true nominal 95th percentile")
    for a in arms:
        ax.plot(t, taus[a], color=COL[a], lw=1.9 if a in ("REV", "NAIVE") else 1.4, alpha=1 if a in ("REV", "LAB", "NAIVE") else 0.8, label=names[a])
    ax.set_xlabel("stream step"); ax.set_ylabel("threshold τ"); ax.set_ylim(-2.4, 6.2)
    ax.set_title("Threshold trajectories (mean of 10 seeds)", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    ax = axs[1]
    ax.axvspan(2000, 3000, color="#EEF1F6", zorder=0)
    ax.axhline(0.05, color="black", lw=1.2, ls=":", label="target α = 0.05")
    for a in arms:
        ax.plot(t, fprs[a], color=COL[a], lw=1.9 if a in ("REV", "NAIVE") else 1.4, alpha=1 if a in ("REV", "LAB", "NAIVE") else 0.8, label=names[a])
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("stream step"); ax.set_ylabel("rolling nominal FPR (400 steps)")
    ax.set_title("What the operator experiences: nuisance alarms", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.045), columnspacing=2.0)
    fig.text(0.5, 0.012, "Illustrative controlled simulation (μ_a = 2.4, π = 5%, W = 200, α = 0.05, γ = 0.01) driving the repository's own "
             "threshold classes in cgc_ad/conformal.py.", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.13, 1, 1]); fig.savefig(out, bbox_inches=None); plt.close(fig)


# ------------------------------------------------------------------ results
def make_arms(rows, out):
    arms = [("LAB", "r3_labelled"), ("REV", "r3_reviewed"), ("ALR", "r3_alarmrate"), ("SL-X", "r3_patch"), ("NAIVE", "r3_naive")]
    fpr = {a: list(class_means(rows, t, "fpr_post").values()) for a, t in arms}
    tpr = {a: list(class_means(rows, t, "tpr_post").values()) for a, t in arms}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axs[0]
    for i, (a, _) in enumerate(arms):
        m = np.mean(fpr[a]); h = t_ci(fpr[a], _T975)
        ax.bar(i, m, color=COL[a], width=0.66, zorder=3)
        ax.errorbar(i, m, yerr=h, color=INK, capsize=4, lw=1.4, zorder=4)
        ax.text(i, m * 1.18 + 0.002, f"{m:.4f}" if m < 0.1 else f"{m:.3f}", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.axhline(0.05, color="black", lw=1.2, ls=":", zorder=5); ax.text(-0.42, 0.057, "target 0.05", ha="left", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(0.006, 2.2); ax.set_xticks(range(len(arms))); ax.set_xticklabels([a for a, _ in arms], fontweight="bold")
    ax.set_ylabel("nominal FPR (log scale)")
    ax.set_title("Only reviewed alarms hit the FPR target\nwithout full labels", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    ax = axs[1]
    for i, (a, _) in enumerate(arms):
        m = np.mean(tpr[a]); h = t_ci(tpr[a], _T975)
        ax.bar(i, m, color=COL[a], width=0.66, zorder=3)
        ax.errorbar(i, m, yerr=h, color=INK, capsize=4, lw=1.4, zorder=4)
        ax.text(i, m + 0.025, f"{m:.3f}", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.12); ax.set_xticks(range(len(arms))); ax.set_xticklabels([a for a, _ in arms], fontweight="bold")
    ax.set_ylabel("TPR (share of defects caught)")
    ax.set_title("REV keeps LAB's detection power; ALR and SL-X pay for\ntheir miscalibration (NAIVE's 0.999 = “flag everything”)", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    fig.text(0.5, -0.04, "MVTec AD · R3-E1 · 5 categories × 5 seeds (class means, 95% t-intervals) · stationary-prevalence stream with mild drift", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def make_burst(rows, out):
    pbs = ["0.05", "0.10", "0.20", "0.40"]
    arms = [("LAB", "labelled"), ("REV", "reviewed"), ("ALR", "alarmrate"), ("SL-X", "patch")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs = np.arange(len(pbs))
    for a, k in arms:
        ys, hs = [], []
        for pb in pbs:
            d = np.array([v for v in paired_burst(rows, f"r3_burst_{k}_{pb}")])
            ys.append(d.mean()); hs.append(t_ci(d, _T975))
        ax.errorbar(xs, ys, yerr=hs, color=COL[a], marker="o", lw=2.4 if a in ("REV", "SL-X") else 1.8, capsize=4, label=a)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(["5%\n(no burst)", "10%", "20%", "40%"])
    ax.set_xlabel("defect prevalence during the 200-step burst"); ax.set_ylabel("change in TPR (during − before)")
    ax.set_title("Exploration lets defects raise the ceiling: SL-X loses detection power", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=4, loc="lower left")
    fig.text(0.5, -0.03, "MVTec AD · R3-E2 · 5 categories × 3 seeds · class means with 95% t-intervals", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def paired_burst(rows, tag):
    a, b = class_means(rows, tag, "tpr_burst"), class_means(rows, tag, "tpr_pre_burst")
    return [a[c] - b[c] for c in sorted(a)]


def make_equivalence(rows, out):
    conds = [("no burst", "r3_reviewed", "r3_labelled")] + [
        (f"burst {int(float(p) * 100)}%" + (" (control)" if p == "0.05" else ""), f"r3_burst_reviewed_{p}", f"r3_burst_labelled_{p}")
        for p in ["0.05", "0.10", "0.20", "0.40"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for i, (name, a, b) in enumerate(conds):
        d = paired(rows, a, b, "fpr_post"); m = d.mean(); h = t_ci(d, _T95)
        ax.errorbar(m, i, xerr=h, fmt="o", color=COL["REV"], capsize=5, lw=2.4, ms=7, zorder=4)
        ax.text(0.0115, i, f"Δ = {m:+.5f}".replace("-", "−"), va="center", fontsize=9.5, color=INK)
    ax.axvspan(-0.01, 0.01, color="#E7F3EA", zorder=0)
    ax.axvline(0, color="black", lw=0.8); ax.axvline(-0.01, color=GREEN, lw=1.4, ls="--"); ax.axvline(0.01, color=GREEN, lw=1.4, ls="--")
    ax.text(0, -0.82, "equivalence margin ±0.01 FPR", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([c[0] for c in conds]); ax.invert_yaxis()
    ax.set_xlim(-0.0125, 0.0225); ax.set_ylim(len(conds) - 0.4, -1.1)
    ax.set_xlabel("REV − LAB nominal FPR (90% CI)")
    ax.set_title("Reviewed alarms are statistically equivalent to full labels", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.04, "TOST, paired by category (n = 5, 4 d.f.); every interval sits 6–18× inside the margin; p < 0.0001 in all five conditions", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def make_naive(rows, out):
    d = class_means(rows, "r3_naive_nodrift", "fpr_post")
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("tag") == "r3_naive_nodrift" and "stream" in r:
            by[r["class"]].append(r["stream"]["fpr_post"])
    cls = sorted(by)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, c in enumerate(cls):
        ax.scatter(np.full(len(by[c]), i) + np.linspace(-0.12, 0.12, len(by[c])), by[c], color=COL["NAIVE"], s=42, zorder=3, alpha=0.9)
        ax.plot([i - 0.25, i + 0.25], [d[c]] * 2, color=INK, lw=2.2, zorder=4)
    ax.axhline(0.05, color="black", lw=1.2, ls=":"); ax.text(len(cls) - 0.55, 0.07, "target α = 0.05", ha="right", fontsize=9)
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, fontweight="bold"); ax.set_ylim(0, 0.85)
    ax.set_ylabel("nominal FPR (post-onset window)")
    allv = np.concatenate([by[c] for c in cls])
    ax.set_title(f"Naive self-labelling with NO drift: mean FPR {allv.mean():.3f}, all {len(allv)} runs above 0.5", fontsize=12, loc="left", fontweight="bold", color=NAVY)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="folder with e1_e3_results.jsonl and e2_e4_results.jsonl")
    ap.add_argument("--out", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--skip-sim", action="store_true", help="skip the (slow) illustrative simulation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    make_banner(os.path.join(a.out, "banner.png"))
    make_censoring(os.path.join(a.out, "censoring.png"))
    make_architecture(os.path.join(a.out, "architecture.png"))
    if not a.skip_sim:
        make_ratchet(os.path.join(a.out, "ratchet_sim.png"))
    res = a.results or next((p for p in (os.path.join(ROOT, "tests", "results"), os.path.join(ROOT, "results")) if os.path.isdir(p)), None)
    if res:
        rows = []
        for f in ("e1_e3_results.jsonl", "e2_e4_results.jsonl"):
            p = os.path.join(res, f)
            if os.path.exists(p):
                rows += load_jsonl(p)
        make_arms(rows, os.path.join(a.out, "results_arms.png"))
        make_burst(rows, os.path.join(a.out, "results_burst.png"))
        make_equivalence(rows, os.path.join(a.out, "results_equivalence.png"))
        make_naive(rows, os.path.join(a.out, "results_naive_nodrift.png"))
    print("assets written to", a.out)


if __name__ == "__main__":
    main()
