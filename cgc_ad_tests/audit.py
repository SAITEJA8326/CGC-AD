"""Audits over results files. Run these BEFORE reading any number.

    python -m cgc_ad.audit phase*.jsonl
    python -m cgc_ad.audit phase4_ablations.jsonl --audit sweep
    python -m cgc_ad.audit *.jsonl --strict          # exit 1 on any FAIL

Unit tests check that the code is correct. Audits check that the *experiment*
is valid -- a distinction that matters, because every defect below was present
in a results file produced by code that passed its tests:

  A1 sweep integrity  the contamination sweep varied `prevalence`, not
                      `contamination`, which stayed 0.00 in all 24 runs, while
                      `trim_eps` tracked the sweep variable in one column.
  A2 completeness     the RQ3 design was missing the `random x decoupled` cell,
                      so an interaction could not be ruled out.
  A3 budget matching  arms must be matched on gradient steps, synthetic budget
                      and adaptation events, or a win is just more compute.
  A4 gates            clean AUROC 0.88 vs ~0.99 for PatchCore; `toothbrush` had
                      n_calib=15, making alpha=0.05 unattainable and tau=inf.
  A5 degeneracy       non-clean buffers posted FPR ~0.99 from buffer starvation.
  A6 statistics       seeds are not independent replicates of a class effect;
                      the multiple-comparison surface must be counted.
  A7 provenance       duplicate hashes, mixed environments, silent failures.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as m_np

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_RESULTS: List[Tuple[str, str, str]] = []
# Arms that are DESIGNED to be degenerate (e.g. r3_naive, the ratchet
# demonstration). Their degeneracy is the result, not a defect, so the
# degeneracy audit reports them as declared instead of failing the file.
_DECLARED: set = set()


def say(status: str, audit: str, msg: str) -> None:
    _RESULTS.append((status, audit, msg))
    mark = {OK: "  ok  ", WARN: " WARN ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {msg}")


def load(paths: Sequence[str]) -> List[Dict]:
    rows, bad = [], 0
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                r["_file"] = p
                rows.append(r)
    if bad:
        say(WARN, "load", f"{bad} unparseable lines skipped")
    return rows


def _get(d: Dict, path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# Fields whose variation defines an experiment. Anything varying outside the
# declared sweep variable is a confound.
SCIENTIFIC_FIELDS = [
    "curriculum.kind", "curriculum.signal", "curriculum.target_miss",
    "curriculum.eta", "curriculum.min_prob",
    "threshold.policy", "threshold.alpha", "threshold.window",
    "threshold.gamma_aci", "threshold.trim_eps", "threshold.aci_update_on",
    "stream.buffer_protocol", "stream.contamination", "stream.admit_explore",
    "stream.prevalence", "stream.couple", "stream.adapt", "stream.adapt_trigger",
    "stream.adapt_steps", "stream.steps",
    "drift.kind", "drift.severity", "drift.onset", "drift.ramp",
    "train.max_steps", "train.batch_size", "train.lr",
    "model.backbone", "model.stages", "model.layers", "model.pretrained",
    "model.proj_dim", "model.feat_size",
    "synth.n_levels", "synth.variants_per_image",
    "data.image_size", "data.n_calib", "data.n_val",
]


# ==========================================================================
# A1 -- sweep integrity
# ==========================================================================

def audit_sweep(rows: List[Dict], expect: Optional[str] = None) -> None:
    print("\n--- A1  sweep integrity " + "-" * 55)
    per_tag = defaultdict(list)
    for r in rows:
        per_tag[r.get("tag", "?")].append(r.get("config", {}))

    varying = {}
    for f in SCIENTIFIC_FIELDS:
        vals = {json.dumps(_get(c, f), sort_keys=True, default=str)
                for cs in per_tag.values() for c in cs}
        if len(vals) > 1:
            varying[f] = sorted(vals)

    if not varying:
        say(FAIL, "sweep", "NO scientific field varies across these rows: "
                           "every run is the same experiment")
        return
    print("  fields that actually vary across the file(s):")
    for f, v in sorted(varying.items()):
        short = ", ".join(v[:6]) + (" ..." if len(v) > 6 else "")
        print(f"    {f:34s} {len(v)} values: {short}")

    if expect:
        if expect in varying:
            say(OK, "sweep", f"declared sweep variable `{expect}` does vary "
                             f"({len(varying[expect])} values)")
            others = [f for f in varying if f != expect]
            if others:
                say(WARN, "sweep",
                    f"{len(others)} OTHER field(s) also vary and may confound "
                    f"the sweep: {', '.join(others[:6])}")
        else:
            say(FAIL, "sweep",
                f"declared sweep variable `{expect}` is CONSTANT across all "
                f"runs. This is the exact defect that invalidated the first "
                f"contamination sweep. Nothing in this file measures it.")

    # Tags that encode a value in their name must match the config.
    mismatch = []
    for tag, cs in per_tag.items():
        parts = tag.replace("-", "_").split("_")
        nums = [p for p in parts if _isnum(p)]
        if not nums:
            continue
        claimed = float(nums[-1])
        hits = [f for f in SCIENTIFIC_FIELDS
                if all(_num(_get(c, f)) == claimed for c in cs)]
        if not hits:
            mismatch.append((tag, claimed))
    if mismatch:
        say(FAIL, "sweep",
            f"{len(mismatch)} tag(s) encode a number no config field matches, "
            f"e.g. {mismatch[:4]}. The tag is lying about what was run.")
    elif any(_isnum(p) for t in per_tag for p in t.split("_")):
        say(OK, "sweep", "numeric tags are consistent with the stored configs")


def _isnum(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
        else None


# ==========================================================================
# A2 -- design completeness
# ==========================================================================

def audit_completeness(rows: List[Dict]) -> None:
    print("\n--- A2  design completeness " + "-" * 51)
    cells = defaultdict(set)
    for r in rows:
        c = r.get("config", {})
        cells[(_get(c, "curriculum.kind"), _get(c, "stream.couple"))].add(
            (r.get("class"), r.get("seed")))
    if len(cells) > 1:
        ctrls = sorted({k[0] for k in cells if k[0] is not None})
        coups = sorted({k[1] for k in cells if k[1] is not None}, key=str)
        missing = [(a, b) for a in ctrls for b in coups if (a, b) not in cells]
        print(f"  controller x couple cells present: {len(cells)} "
              f"of {len(ctrls) * len(coups)}")
        if missing:
            say(WARN, "design",
                f"missing cell(s) {missing}. Without the full factorial you "
                f"cannot separate a main effect from an interaction -- this is "
                f"why `random x decoupled` had to be run after the fact.")
        else:
            say(OK, "design", "controller x couple factorial is complete")

    per_arm = defaultdict(lambda: defaultdict(set))
    for r in rows:
        per_arm[r.get("tag")][r.get("class")].add(r.get("seed"))
    sizes = {t: (len(d), sorted({len(s) for s in d.values()}))
             for t, d in per_arm.items()}
    ncls = {n for n, _ in sizes.values()}
    if len(ncls) > 1:
        say(WARN, "design",
            f"arms cover different numbers of classes {sorted(ncls)}; paired "
            f"tests silently drop the non-overlapping ones")
    for t, (n, seedcounts) in sorted(sizes.items()):
        if len(seedcounts) > 1:
            say(WARN, "design",
                f"arm `{t}` has ragged seed counts {seedcounts} across classes; "
                f"class means are estimated with unequal precision")


# ==========================================================================
# A3 -- budget matching
# ==========================================================================

def audit_budget(rows: List[Dict]) -> None:
    print("\n--- A3  budget matching " + "-" * 55)
    fields = {"gradient steps": lambda r: r.get("steps"),
              "adaptation events": lambda r: _get(r, "stream.n_adaptations"),
              "trainable params": lambda r: r.get("trainable_params"),
              "synth variants": lambda r: _get(r, "config.synth.variants_per_image"),
              "adapt_steps": lambda r: _get(r, "config.stream.adapt_steps")}
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for name, fn in fields.items():
            v = fn(r)
            if isinstance(v, (int, float)):
                per[name][r.get("tag")].append(float(v))

    for name, d in per.items():
        means = {t: m_np.mean(v) for t, v in d.items()}
        if len(means) < 2:
            continue
        lo, hi = min(means.values()), max(means.values())
        spread = (hi - lo) / max(abs(hi), 1e-9)
        line = "  " + name.ljust(20) + " ".join(
            f"{t}={v:.2f}" for t, v in sorted(means.items()))
        print(line[:160])
        if spread > 0.10:
            say(FAIL, "budget",
                f"`{name}` differs by {spread:.0%} across arms. Any performance "
                f"gap is confounded with compute; match it or report it.")
        elif spread > 0.02:
            say(WARN, "budget", f"`{name}` differs by {spread:.1%} across arms")
        else:
            say(OK, "budget", f"`{name}` matched across arms ({spread:.1%})")


# ==========================================================================
# A4 -- gates
# ==========================================================================

def audit_gates(rows: List[Dict]) -> None:
    print("\n--- A4  gates " + "-" * 65)
    mono = [(r.get("class"), _get(r, "monotonicity.spearman"),
             _get(r, "monotonicity.range")) for r in rows
            if _get(r, "monotonicity.spearman") is not None]
    if mono:
        per = defaultdict(list)
        for c, s, g in mono:
            per[c].append(s > 0.8 and g > 0.3)
        rate = m_np.mean([v for vs in per.values() for v in vs])
        bad = sorted([(c, float(m_np.mean(v))) for c, v in per.items()
                      if m_np.mean(v) < 0.8], key=lambda x: x[1])
        (say(OK, "gate", f"monotonicity: {rate:.0%} of runs pass")
         if rate > 0.9 else
         say(WARN, "gate",
             f"monotonicity: only {rate:.0%} of runs pass. Failing classes have "
             f"an ill-posed hardness axis and a setpoint controller cannot work "
             f"there: {', '.join(f'{c}({p:.2f})' for c, p in bad[:6])}"))

    per_tag = defaultdict(list)
    for r in rows:
        a = _get(r, "static_clean.image_auroc")
        if a is not None:
            per_tag[r.get("tag")].append(a)
    for t, v in sorted(per_tag.items()):
        mu = float(m_np.mean(v))
        (say(OK, "gate", f"clean AUROC [{t}] = {mu:.4f}") if mu >= 0.95 else
         say(FAIL, "gate",
             f"clean AUROC [{t}] = {mu:.4f}, below parity with CutPaste (~0.95) "
             f"and PatchCore (~0.99). Results from an undertrained scorer are "
             f"not interpretable; raise train.max_steps / backbone first."))

    seen = {}
    for r in rows:
        seen.setdefault(r.get("class"), r)
    for c, r in sorted(seen.items()):
        n = _get(r, "data.n_calib")
        a = _get(r, "config.threshold.alpha")
        if n and a and n < 1.0 / a - 1:
            say(FAIL, "gate",
                f"class `{c}`: n_calib={n} cannot express alpha={a} "
                f"(needs >= {math.ceil(1/a - 1)}). Its threshold is +inf and the "
                f"detector never fires. Drop the class or raise alpha.")

    ret = [r.get("auroc_retention") for r in rows
           if isinstance(r.get("auroc_retention"), (int, float))]
    if ret:
        mu = float(m_np.mean(ret))
        (say(OK, "gate", f"AUROC retention under drift = {mu:+.3f}")
         if mu > -0.15 else
         say(WARN, "gate",
             f"AUROC retention = {mu:+.3f}: the drift regime is destroying the "
             f"detector rather than degrading it. Threshold comparisons made on "
             f"a broken scorer say little. Lower drift.severity."))


# ==========================================================================
# A5 -- degeneracy / starvation
# ==========================================================================

def audit_degeneracy(rows: List[Dict]) -> None:
    print("\n--- A5  degeneracy and starvation " + "-" * 45)
    bad, ratchet = [], []
    declared = [r for r in rows if r.get("tag") in _DECLARED]
    if declared:
        say(OK, "degeneracy",
            f"{len(declared)} run(s) in declared degeneracy ablation(s) "
            f"{sorted({r.get('tag') for r in declared})} -- excluded from the "
            f"checks below and expected to fail them")
    rows = [r for r in rows if r.get("tag") not in _DECLARED]
    for r in rows:
        fpr = _get(r, "stream.fpr_post")
        proto = _get(r, "config.stream.buffer_protocol")
        upd = _get(r, "config.threshold.aci_update_on", "admitted")
        expl = _get(r, "config.stream.admit_explore", 0.0) or 0.0
        if fpr is not None and fpr > 0.5:
            bad.append((r.get("tag"), r.get("class"), fpr, proto))
        if proto in ("self_label", "gated") and (upd == "admitted" or expl <= 0):
            ratchet.append((r.get("tag"), proto, upd, expl))

    if bad:
        say(FAIL, "degeneracy",
            f"{len(bad)} run(s) have post-drift FPR > 0.5. The buffer has "
            f"starved: the quantile is being estimated from stale censored "
            f"data and EVERY other metric in those rows is meaningless. "
            f"e.g. {bad[:3]}")
    else:
        say(OK, "degeneracy", "no run shows runaway false-alarm rates")

    if ratchet:
        say(FAIL, "degeneracy",
            f"{len(ratchet)} run(s) use a self-labelled/gated buffer with "
            f"aci_update_on='admitted' and/or admit_explore=0. Then the ACI "
            f"error indicator is identically zero AND tau is a monotone "
            f"non-increasing ratchet -- it cannot track upward drift by "
            f"construction. Valid only as the declared degeneracy ablation.")

    pre = [(_get(r, "stream.mean_tau_pre"), _get(r, "stream.mean_tau_post"))
           for r in rows]
    infs = sum(1 for a, b in pre if a == float("inf") or b == float("inf"))
    if infs:
        say(FAIL, "degeneracy",
            f"{infs} run(s) report tau = +inf: an unattainable alpha, so the "
            f"detector never fires while its FPR log looks perfect.")


# ==========================================================================
# A6 -- statistical validity
# ==========================================================================

def audit_statistics(rows: List[Dict]) -> None:
    print("\n--- A6  statistical validity " + "-" * 50)
    classes = {r.get("class") for r in rows}
    seeds = {r.get("seed") for r in rows}
    arms = {r.get("tag") for r in rows}
    print(f"  classes={len(classes)}  seeds={len(seeds)}  arms={len(arms)}")

    if len(classes) < 8:
        say(WARN, "stats",
            f"only {len(classes)} classes. Classes are the unit of analysis, so "
            f"a Wilcoxon signed-rank test here has very little power; report "
            f"effect sizes and CIs rather than p-values.")
    else:
        say(OK, "stats", f"{len(classes)} classes is enough for a paired test")

    if len(seeds) < 3:
        say(WARN, "stats", f"only {len(seeds)} seed(s): class means are noisy")

    n_metrics = 7
    n_contrasts = max(0, len(arms) - 1)
    total = n_metrics * n_contrasts
    if total > 10:
        say(WARN, "stats",
            f"~{total} hypotheses implied ({n_contrasts} contrasts x ~{n_metrics} "
            f"metrics). Declare ONE primary endpoint in advance and apply Holm "
            f"within each family, or the false-discovery surface swamps the "
            f"effect sizes you are reporting.")

    dup = Counter((r.get("run_hash"), r.get("class"), r.get("seed"))
                  for r in rows)
    rep = [k for k, v in dup.items() if v > 1]
    if rep:
        say(FAIL, "stats",
            f"{len(rep)} duplicated (hash, class, seed) triple(s). These would "
            f"be double-counted in class means. e.g. {rep[:3]}")
    else:
        say(OK, "stats", "no duplicated (hash, class, seed) triples")

    collide = defaultdict(set)
    for r in rows:
        collide[r.get("run_hash")].add(r.get("tag"))
    bad = {h: t for h, t in collide.items() if len(t) > 1}
    if bad:
        say(FAIL, "stats",
            f"{len(bad)} run hash(es) shared by multiple tags: two different "
            f"experiments are colliding and --resume will have skipped one. "
            f"e.g. {list(bad.items())[:2]}")


# ==========================================================================
# A7 -- provenance
# ==========================================================================

def audit_feedback_semantics(rows: List[Dict]) -> None:
    """An arm tagged oracle / clean / labelled must regulate nominal FPR.

    WHY: changing the default of `threshold.aci_update_on` from nominal-only
    feedback to all-sample feedback silently converted the clean-buffer
    "oracle" arm from labelled ACI into alarm-rate control. Its FPR then sat
    at (alpha - pi*TPR)/(1 - pi) instead of alpha, and the paper reported a
    method as matching a label-supervised baseline that was not label
    supervised. This audit makes that mislabelling impossible.
    """
    bad = []
    for r in rows:
        tag = str(r.get("tag", "")).lower()
        proto = get(r, "config.stream.buffer_protocol")
        upd = get(r, "config.threshold.aci_update_on") or "all"
        if any(k in tag for k in ("oracle", "clean", "label")) and upd == "all":
            bad.append((r.get("tag"), proto, upd))
    if bad:
        say(FAIL, "feedback-semantics",
            f"{len(set(bad))} arm(s) are tagged as labelled/oracle but feed ACI "
            f"the alarm on ALL samples, so they regulate the alarm rate, not "
            f"nominal FPR: {sorted(set(bad))[:4]}. Set "
            f"threshold.aci_update_on=admitted with buffer_protocol=clean for "
            f"a genuinely labelled baseline.")
    elif rows:
        say(OK, "feedback-semantics", "no arm is mislabelled as label-supervised")


def audit_confounds(rows: List[Dict]) -> None:
    """Design confounds that are invisible in any single number.

    WHY: in round 3 every arm ran with stream.adapt=true (the breakdown.yaml
    default). Adaptation events fine-tune the head and refit tau, so the
    "static" arm was recalibrated 1.24 times per run on average and was never
    a fixed threshold, and arms differed in how often their head was retrained.
    Separately, brightness drift at severity 0.25 moved clean AUROC by -0.001:
    those runs were effectively stationary while being described as drifted.
    """
    print("\n--- A8  design confounds " + "-" * 54)
    st = [r for r in rows if _get(r, "config.threshold.policy") == "static"
          and _get(r, "config.stream.adapt")]
    if st:
        ev = [(_get(r, "stream.n_adaptations") or 0) for r in st]
        say(FAIL, "confound",
            f"{len(st)} static-threshold run(s) have stream.adapt=true "
            f"(mean {float(m_np.mean(ev)):.2f} adaptation events). The head is "
            f"fine-tuned mid-stream, so this is not a static baseline. Rerun "
            f"with stream.adapt=false.")
    ad = [r for r in rows if _get(r, "config.stream.adapt")]
    if ad and len(ad) < len(rows) or (ad and len({_get(r, "stream.n_adaptations")
                                                  for r in ad}) > 1):
        say(WARN, "confound",
            f"{len(ad)} run(s) adapt the head mid-stream and realised events "
            f"differ across arms; threshold comparisons then also compare "
            f"different heads. For threshold studies set stream.adapt=false.")
    by = defaultdict(list)
    for r in rows:
        k = (_get(r, "config.drift.kind"), _get(r, "config.drift.severity"))
        if k[0] not in (None, "none") and isinstance(r.get("auroc_retention"),
                                                     (int, float)):
            by[k].append(r["auroc_retention"])
    for (kind, sev), v in sorted(by.items(), key=lambda x: str(x[0])):
        mu = float(m_np.mean(v))
        if abs(mu) < 0.01:
            say(WARN, "confound",
                f"drift {kind}@{sev}: mean AUROC retention {mu:+.4f}. The "
                f"scorer barely notices this drift, so these runs are "
                f"effectively stationary; do not describe them as drifted.")
        else:
            say(OK, "confound", f"drift {kind}@{sev}: retention {mu:+.4f}")
    if not st:
        say(OK, "confound", "no static-threshold run adapts mid-stream")


def audit_provenance(rows: List[Dict]) -> None:
    print("\n--- A7  provenance " + "-" * 60)
    errs = [r for r in rows if "error" in r]
    if errs:
        say(WARN, "provenance",
            f"{len(errs)} failed run(s) recorded. Silent failures shrink arms "
            f"unevenly and break pairing. e.g. "
            f"{[(e.get('tag'), e.get('class')) for e in errs[:3]]}")
    else:
        say(OK, "provenance", "no recorded run failures")

    envs = {json.dumps({k: _get(r, f"env.{k}") for k in
                        ("torch", "cuda", "gpu")}, sort_keys=True)
            for r in rows if r.get("env")}
    if len(envs) > 1:
        say(WARN, "provenance",
            f"{len(envs)} distinct environments across these rows; note the "
            f"split in the reproducibility section")
    elif envs:
        say(OK, "provenance", f"single environment: {list(envs)[0]}")
    else:
        say(WARN, "provenance",
            "no environment capture in these rows (older schema); rerun with "
            "the current runner so the paper can state hardware and versions")

    # Duplicate (run_hash, class, seed) triples mean --resume did not skip
    # finished work. Every duplicate is double-weighted by the seed average
    # inside analyze.py, which silently reweights the paired test.
    dup = Counter((r.get("run_hash"), r.get("class"), r.get("seed"))
                  for r in rows if r.get("run_hash"))
    repeated = {k: v for k, v in dup.items() if v > 1}
    if repeated:
        say(FAIL, "provenance",
            f"{len(repeated)} (run_hash, class, seed) triple(s) appear more "
            f"than once, e.g. {list(repeated.items())[:3]}. --resume should "
            f"have skipped these; seed averaging will double-count them.")
    elif dup:
        say(OK, "provenance", f"{len(dup)} unique (hash, class, seed) triples")

    missing = [k for k in ("stream", "static_clean", "monotonicity")
               if not any(k in r for r in rows)]
    if missing:
        say(WARN, "provenance",
            f"no row contains {missing} -- these were probably --stage gate or "
            f"--stage train runs; do not mix them into stream comparisons")


AUDITS = {
    "feedback": audit_feedback_semantics,
    "confounds": audit_confounds,
    "sweep": audit_sweep, "completeness": audit_completeness,
          "budget": audit_budget, "gates": audit_gates,
          "degeneracy": audit_degeneracy, "statistics": audit_statistics,
          "provenance": audit_provenance}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("cgc-ad-audit")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--audit", default="all", choices=["all", *AUDITS])
    ap.add_argument("--expect-sweep", default=None,
                    help="dotted config field this file is supposed to sweep, "
                         "e.g. stream.contamination")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any FAIL")
    ap.add_argument("--declared", nargs="*", default=[],
                    help="tags of arms that are deliberately degenerate "
                         "ablations, e.g. --declared r3_naive r4_naive_nodrift")
    args = ap.parse_args(argv)

    _DECLARED.clear()
    _DECLARED.update(args.declared)
    rows = load(args.paths)
    print(f"\n{len(rows)} rows from {len(args.paths)} file(s) | "
          f"tags={sorted({r.get('tag') for r in rows})}")

    for name, fn in AUDITS.items():
        if args.audit in ("all", name):
            fn(rows, args.expect_sweep) if name == "sweep" else fn(rows)

    n_fail = sum(1 for s, _, _ in _RESULTS if s == FAIL)
    n_warn = sum(1 for s, _, _ in _RESULTS if s == WARN)
    print(f"\n{'=' * 78}\nAUDIT SUMMARY: {n_fail} FAIL, {n_warn} WARN, "
          f"{sum(1 for s, _, _ in _RESULTS if s == OK)} PASS")
    if n_fail:
        print("Do not report numbers from these files until the FAILs are "
              "resolved or explicitly justified in the paper.")
    return 1 if (args.strict and n_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
