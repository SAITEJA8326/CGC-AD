"""Tests for the audit module itself.

WHY THIS FILE EXISTS
The audits are the last line of defence between a defective experiment and a
table in the paper. An audit that cannot detect the defect it was written for
is worse than no audit at all, because it manufactures confidence. Each test
below injects the *exact* defect that occurred in a real results file and
asserts the audit reports FAIL.

Unit tests verify the code. These verify the verifier.
"""
import numpy as np
import pytest

from cgc_ad.audit import (FAIL, OK, WARN, _RESULTS, audit_budget,
                          audit_completeness, audit_degeneracy, audit_gates,
                          audit_provenance, audit_statistics, audit_sweep)


def _row(tag="a", cls="bottle", seed=0, **ov):
    """A well-formed result row; keyword overrides use dotted config paths."""
    cfg = {
        "stream": {"contamination": 0.0, "prevalence": 0.05, "couple": True,
                   "buffer_protocol": "clean", "adapt": True, "adapt_steps": 100,
                   "admit_explore": 0.05, "steps": 1500},
        "threshold": {"policy": "aci", "alpha": 0.05, "window": 200,
                      "trim_eps": 0.05, "gamma_aci": 0.01,
                      "aci_update_on": "all"},
        "curriculum": {"kind": "proportional", "signal": "conformal",
                       "target_miss": 0.15},
        "drift": {"kind": "brightness", "severity": 0.25, "onset": 500},
        "synth": {"n_levels": 6, "variants_per_image": 2},
        "model": {"backbone": "wide_resnet50_2", "proj_dim": 128, "feat_size": 28},
        "train": {"max_steps": 8000, "batch_size": 32},
        "data": {"image_size": 224, "n_calib": 60, "n_val": 30},
    }
    for k, v in ov.items():
        sec, _, key = k.partition(".")
        cfg.setdefault(sec, {})[key] = v
    return {
        "tag": tag, "class": cls, "seed": seed, "config": cfg,
        "run_hash": "abc123", "steps": 8000, "trainable_params": 100_000,
        "data": {"n_train": 200, "n_calib": 60, "n_val": 30},
        "static_clean": {"image_auroc": 0.962},
        "static_drift": {"image_auroc": 0.905},
        "auroc_retention": -0.057,
        "monotonicity": {"spearman": 0.98, "range": 0.55,
                         "levels": [0, .2, .4, .6, .8, 1.0],
                         "miss_rate": [.02, .1, .2, .35, .5, .6]},
        "env": {"torch": "2.4.0"},
        "stream": {"n_adaptations": 4.0, "fpr_post": 0.052, "tpr_post": 0.71,
                   "mad_fpr": 0.021, "mean_tau_pre": -2.0,
                   "mean_tau_post": -1.6, "stream_auroc_post": 0.84,
                   "recovery_delay": 12.0},
    }


@pytest.fixture(autouse=True)
def _clear():
    _RESULTS.clear()
    yield


def _statuses():
    return [s for s, _, _ in _RESULTS]


def _messages():
    return " | ".join(m for _, _, m in _RESULTS)


# ------------------------------------------------------------------ A1 sweep

def test_sweep_audit_fails_when_the_declared_variable_never_moved():
    """THE PHASE-5 DEFECT, reproduced exactly.

    24 runs were tagged as a contamination sweep. `stream.contamination` was
    0.00 in every one of them; what actually moved was `stream.prevalence`.
    The audit must FAIL rather than quietly report a flat curve.
    """
    rows = [_row(tag=f"bd_clean_{p}", **{"stream.prevalence": p})
            for p in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]]
    audit_sweep(rows, expect="stream.contamination")
    assert FAIL in _statuses()
    assert "CONSTANT" in _messages()


def test_sweep_audit_passes_and_flags_confounds_on_a_real_sweep():
    """A correct sweep must PASS on the declared variable, and any *other*
    field that also varies must be surfaced as a potential confound -- that is
    how `trim_eps` moving in lockstep with the sweep index gets caught."""
    rows = [_row(tag=f"bd_{e}", **{"stream.contamination": e,
                                   "threshold.trim_eps": e})
            for e in [0.0, 0.01, 0.05, 0.20]]
    audit_sweep(rows, expect="stream.contamination")
    assert OK in _statuses()
    assert WARN in _statuses()


def test_sweep_audit_catches_a_tag_that_lies_about_its_config():
    """Tags like `bd_selflabel_0.10` encode a number. If no config field equals
    that number, the tag is describing a run that did not happen -- a mislabel
    that survives every other check and poisons the grouping in every figure."""
    # something must vary, or the audit short-circuits before the tag check
    rows = [_row(tag="bd_clean_0.10", **{"stream.contamination": 0.0,
                                         "threshold.window": 100}),
            _row(tag="bd_clean_0.20", **{"stream.contamination": 0.0,
                                         "threshold.window": 200})]
    audit_sweep(rows)
    assert FAIL in _statuses()
    assert "lying" in _messages()


# ----------------------------------------------------------- A2 completeness

def test_completeness_audit_notices_a_missing_design_cell():
    """The RQ3 design needs all four of {proportional, random} x {coupled,
    decoupled}. With `random x decoupled` absent, the coupling effect cannot be
    separated from a controller interaction, and the audit should say so before
    the contrast reaches a table."""
    rows = []
    for tag in ["coupled", "decoupled", "ctrl_random"]:
        for c in ["bottle", "carpet", "tile"]:
            for s in range(3):
                rows.append(_row(tag=tag, cls=c, seed=s))
    # ctrl_random is short one class -> a ragged grid
    rows = [r for r in rows if not (r["tag"] == "ctrl_random" and r["class"] == "tile")]
    audit_completeness(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())


def test_completeness_audit_passes_a_full_grid():
    rows = [_row(tag=t, cls=c, seed=s)
            for t in ["coupled", "decoupled"]
            for c in ["bottle", "carpet", "tile"] for s in range(3)]
    audit_completeness(rows)
    assert FAIL not in _statuses()


# ----------------------------------------------------------------- A3 budget

def test_budget_audit_flags_unmatched_gradient_steps():
    """'You just trained longer' is the first objection any adaptive method
    receives. It has to be refuted from the logs, not asserted in prose."""
    rows = [_row(tag="a"), _row(tag="b")]
    for r in rows[1:]:
        r["steps"] = 16000
        r["config"]["train"]["max_steps"] = 16000
    audit_budget(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())


def test_budget_audit_passes_matched_arms():
    rows = [_row(tag="coupled"), _row(tag="decoupled",
                                      **{"stream.couple": False})]
    audit_budget(rows)
    assert FAIL not in _statuses()


# ------------------------------------------------------------------ A4 gates

def test_gates_audit_catches_sub_parity_auroc():
    """Clean AUROC of 0.88 against PatchCore's ~0.99 means every comparative
    claim is being made in a weak-detector regime. That has to be a FAIL, not
    a footnote."""
    rows = [_row(cls=c) for c in ["bottle", "carpet", "tile"]]
    for r in rows:
        r["static_clean"]["image_auroc"] = 0.88
    audit_gates(rows)
    assert FAIL in _statuses()


def test_gates_audit_catches_unattainable_alpha():
    """toothbrush: n_calib=15 cannot express alpha=0.05, tau comes back +inf,
    the detector never fires, and the FPR log reads a spotless 0.000. The most
    dangerous failure in the system, because it looks like success."""
    r = _row(cls="toothbrush")
    r["data"]["n_calib"] = 15
    r["config"]["data"]["n_calib"] = 15
    r["stream"]["mean_tau_post"] = float("inf")
    audit_gates([r])
    assert FAIL in _statuses()


def test_gates_audit_catches_a_flat_hardness_axis():
    """A monotonicity range of 0.000 means every hardness level produces the
    same miss-rate, so a setpoint controller has nothing to control."""
    rows = [_row(cls=c) for c in ["a", "b", "c"]]
    for r in rows:
        r["monotonicity"] = {"spearman": 0.1, "range": 0.0,
                             "levels": [0, .5, 1.], "miss_rate": [.3, .3, .3]}
    audit_gates(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())


# ------------------------------------------------------------- A5 degeneracy

def test_degeneracy_audit_identifies_the_self_labelling_ratchet():
    """Post-drift FPR near 1.0 on a non-clean buffer, with tau having moved
    DOWN while drift moved scores UP. The audit must name the ratchet rather
    than leave it looking like a tuning problem."""
    rows = [_row(cls=c, **{"stream.buffer_protocol": "self_label",
                           "threshold.aci_update_on": "admitted",
                           "stream.admit_explore": 0.0})
            for c in ["bottle", "carpet"]]
    for r in rows:
        r["stream"].update({"fpr_post": 0.993, "mean_tau_pre": -6.56,
                            "mean_tau_post": -6.95, "mad_fpr": 0.917})
    audit_degeneracy(rows)
    assert FAIL in _statuses()


def test_degeneracy_audit_passes_a_healthy_clean_buffer():
    audit_degeneracy([_row(cls=c) for c in ["bottle", "carpet"]])
    assert FAIL not in _statuses()


# ------------------------------------------------------------- A6 statistics

def test_statistics_audit_notices_too_few_seeds_or_classes():
    """A paired Wilcoxon over 3 classes cannot reach p<0.05 no matter what the
    effect is. Reporting one is a category error, and the audit should catch it
    before the table is written."""
    rows = [_row(tag=t, cls=c, seed=0)
            for t in ["coupled", "decoupled"] for c in ["bottle", "carpet"]]
    audit_statistics(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())


def test_statistics_audit_notices_nan_metrics():
    """A NaN in tpr_post silently drops that class from a paired test and
    changes the n of the Wilcoxon without telling anyone."""
    rows = [_row(tag=t, cls=c, seed=s)
            for t in ["coupled", "decoupled"]
            for c in ["a", "b", "c", "d", "e", "f"] for s in range(5)]
    for r in rows[:12]:
        r["stream"]["tpr_post"] = float("nan")
    audit_statistics(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())


# ------------------------------------------------------------- A7 provenance

def test_provenance_audit_flags_duplicate_rows():
    """--resume should skip finished work. Duplicates mean it did not, and a
    duplicated (hash, class, seed) is double-weighted by every seed average."""
    rows = [_row(tag="coupled", cls="bottle", seed=0) for _ in range(3)]
    audit_provenance(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())


def test_provenance_audit_flags_mixed_environments():
    """Runs spanning two torch versions are not pooled by default in a paper
    without a sentence justifying it."""
    rows = [_row(cls="a"), _row(cls="b")]
    rows[1]["env"] = {"torch": "2.1.0"}
    audit_provenance(rows)
    assert any(s in (FAIL, WARN) for s in _statuses())
