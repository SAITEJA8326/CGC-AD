"""Regression tests for defects found in the round-3 results.

WHY THIS FILE EXISTS

Each test reproduces a defect that was visible in the R3 outputs and would
otherwise have gone into the paper:

  * E1 audit: five arms at clean AUROC 0.9830, r3_naive alone at 0.9846. The
    head depended on whether the feature bank was built or loaded from cache,
    because synthesis and training shared one RNG. Reproduced: max|dW| = 7.6e-3
    between the first and second arm of a session, 0.0 between later arms.
  * r3_static logged 1.24 adaptation events per run: adaptation refit tau even
    for the static policy, so the static baseline was never static.
  * E4's "s1.2" arm: severity was silently clipped to 1.0.
  * Every 5/5 contrast reported p = 0.059, the anti-conservative normal
    approximation of an exact floor of 0.0625.
"""
import numpy as np
import pytest

from cgc_ad.config import Config

torch = pytest.importorskip("torch")


# ----------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def cd():
    from cgc_ad.data import ClassData
    g = torch.Generator().manual_seed(0)
    S = 224
    mk = lambda n: torch.rand(n, 3, S, S, generator=g) * 0.5 + 0.25
    te = mk(12)
    lab = torch.tensor([0] * 8 + [1] * 4)
    msk = torch.zeros(12, 1, S, S)
    for i in range(8, 12):
        te[i, :, 60:120, 60:120] = 1.0
        msk[i, :, 60:120, 60:120] = 1.0
    return ClassData("fake", mk(24), mk(20), mk(10), te, lab, msk)


def _cfg(tmp, bank_cache=True, head_cache=False, **kw):
    c = Config()
    c.model.backbone = "resnet18"
    c.model.pretrained = False
    c.synth.n_levels = 3
    c.synth.variants_per_image = 1
    c.train.max_steps = 15
    c.train.batch_size = 8
    c.train.amp = False
    c.ckpt.enabled = False
    c.ckpt.save_best = False
    c.ckpt.cache_banks = bank_cache
    c.ckpt.head_cache = head_cache
    c.ckpt.banks_dir = str(tmp)
    c.ckpt.resume = True
    for k, v in kw.items():
        sec, _, key = k.partition("__")
        setattr(getattr(c, sec), key, v)
    return c


def _w(head):
    return torch.cat([p.detach().flatten() for p in head.parameters()])


# ------------------------------------------------------------ head identity

def test_head_does_not_depend_on_bank_cache_or_torch_history(cd, tmp_path):
    """THE E1 BUG. The trained head must be a function of (config, class, seed)
    only -- not of whether the bank was cached, nor of how much of torch's global
    RNG earlier classes consumed."""
    from cgc_ad.run import prepare_head, set_seed
    heads = []
    for bank_cache, burn in [(True, 0), (True, 0), (False, 0), (True, 777)]:
        set_seed(0)
        torch.rand(burn)
        *_, h, _, _ = prepare_head(_cfg(tmp_path, bank_cache), cd, "cpu")
        heads.append(_w(h))
    for h in heads[1:]:
        assert torch.equal(h, heads[0])


def test_head_cache_hit_is_bit_identical(cd, tmp_path):
    from cgc_ad.run import prepare_head
    *_, h1, t1, _ = prepare_head(_cfg(tmp_path, head_cache=True), cd, "cpu")
    *_, h2, t2, _ = prepare_head(_cfg(tmp_path, head_cache=True), cd, "cpu")
    assert (t1["head_cache"], t2["head_cache"]) == ("miss", "hit")
    assert torch.equal(_w(h1), _w(h2))


def test_head_key_ignores_stream_but_not_training(tmp_path):
    """Arms that differ only in the threshold layer must share a head; any
    change to training must not."""
    from cgc_ad.checkpoint import head_key
    base = head_key(_cfg(tmp_path), "bottle")
    for k, v in [("stream__buffer_protocol", "reviewed"),
                 ("threshold__aci_update_on", "all"),
                 ("stream__admit_explore", 0.05), ("drift__severity", 0.9),
                 ("stream__adapt", False), ("threshold__policy", "static")]:
        assert head_key(_cfg(tmp_path, **{k: v}), "bottle") == base, k
    for k, v in [("train__max_steps", 16), ("curriculum__kind", "fixed"),
                 ("model__head_width", 64), ("threshold__alpha", 0.1)]:
        assert head_key(_cfg(tmp_path, **{k: v}), "bottle") != base, k


# ------------------------------------------------------------ stream identity

def _stream(cd, tmp, **kw):
    from cgc_ad.engine import run_stream
    from cgc_ad.run import prepare_head, set_seed
    set_seed(0)
    opts = dict(stream__steps=160, drift__kind="none", drift__severity=0.0,
                drift__onset=40, stream__adapt=False)
    opts.update(kw)
    c = _cfg(tmp, head_cache=True, **opts)
    bb, banks, ctrl, head, _, streams = prepare_head(c, cd, "cpu")
    m, rec = run_stream(head, bb, cd, banks, ctrl, c, "cpu", streams["stream"])
    return m, rec


def test_stream_is_identical_whether_head_was_cached_or_trained(cd, tmp_path):
    """Admission randomness (exploration, contamination) must come from its own
    stream, or a head-cache hit would change which samples are admitted."""
    kw = dict(stream__buffer_protocol="self_label", stream__admit_explore=0.2,
              threshold__aci_update_on="all")
    _, a = _stream(cd, tmp_path, **kw)            # miss
    _, b = _stream(cd, tmp_path, **kw)            # hit
    assert [r["admitted"] for r in a] == [r["admitted"] for r in b]
    assert [r["tau"] for r in a] == [r["tau"] for r in b]


def test_arms_share_the_stream_and_the_head(cd, tmp_path):
    """The paired design: two arms see the same samples in the same order and
    score them with the same head, so any difference is the threshold layer."""
    _, a = _stream(cd, tmp_path, threshold__aci_update_on="admitted")
    _, b = _stream(cd, tmp_path, stream__buffer_protocol="reviewed")
    assert [r["label"] for r in a] == [r["label"] for r in b]
    assert [r["score"] for r in a] == [r["score"] for r in b]


def test_stream_logs_prevalence_and_identity_prediction(cd, tmp_path):
    m, rec = _stream(cd, tmp_path)
    post = [r for r in rec if r["t"] >= 40]
    assert m["prevalence_post"] == pytest.approx(np.mean([r["label"] for r in post]))
    if np.isfinite(m["tpr_post"]):
        a, pi = 0.05, m["prevalence_post"]
        assert m["fpr_alarm_identity"] == pytest.approx((a - pi * m["tpr_post"]) / (1 - pi))


# ------------------------------------------------------------ static policy

def test_static_threshold_is_never_refit_by_adaptation(cd, tmp_path):
    """r3_static logged 1.24 adaptation events per run and had its tau
    rewritten at each. A static threshold must stay at tau_0 for the whole
    stream, adaptation or not."""
    m, rec = _stream(cd, tmp_path, threshold__policy="static", stream__adapt=True,
                     stream__adapt_trigger="timer", stream__adapt_period=40,
                     stream__adapt_steps=3, stream__couple=False)
    taus = {r["tau"] for r in rec}
    assert len(taus) == 1, f"static tau changed: {sorted(taus)[:4]}"


# ------------------------------------------------------------------ severity

def test_severity_above_one_is_not_silently_clipped():
    from cgc_ad.data import apply_drift
    x = torch.full((1, 3, 64, 64), 0.4) + 0.1 * torch.rand(
        1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    for k in ["brightness", "gamma", "contrast", "blur", "noise", "combo"]:
        a = apply_drift(x, k, 1.0, torch.Generator().manual_seed(0))
        b = apply_drift(x, k, 1.2, torch.Generator().manual_seed(0))
        assert not torch.equal(a, b), k


def test_severity_beyond_two_is_rejected():
    from cgc_ad.data import apply_drift
    with pytest.raises(ValueError):
        apply_drift(torch.rand(1, 3, 64, 64), "gamma", 2.5)


# ------------------------------------------------------------------ statistics

def test_exact_wilcoxon_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    from cgc_ad.analyze import wilcoxon_p
    rng = np.random.default_rng(0)
    for n in range(5, 16):
        for _ in range(10):
            d = rng.normal(0.3, 1.0, n)
            assert wilcoxon_p(d) == pytest.approx(
                scipy_stats.wilcoxon(d, method="exact").pvalue, abs=1e-12)


def test_wilcoxon_floor_is_reported_correctly():
    """At n = 5 no contrast can reach p < 0.05. The normal approximation
    returned 0.059 here; the exact value is 2/2^5."""
    from cgc_ad.analyze import wilcoxon_p, wilcoxon_floor
    assert wilcoxon_p(np.array([.1, .2, .3, .4, .5])) == pytest.approx(0.0625)
    assert wilcoxon_floor(5) >= 0.05
    assert wilcoxon_p(np.arange(1, 15) / 10) == pytest.approx(2 / 2 ** 14)


def test_wilcoxon_handles_zeros_instead_of_returning_nan():
    from cgc_ad.analyze import wilcoxon_p
    p = wilcoxon_p(np.array([0.01, 0.0, 0.0, 0.02, -0.005]))
    assert np.isfinite(p) and 0 < p <= 1


def test_identity_fallback_recovers_prevalence_from_logged_rates():
    """Rows written before prevalence_post existed can still be checked:
    alarm = (1-pi) f + pi t  =>  pi = (alarm - f)/(t - f)."""
    from cgc_ad.analyze import _identity
    a, pi, t = 0.05, 0.05, 0.78
    f = (a - pi * t) / (1 - pi)
    row = {"config": {"threshold": {"alpha": a}},
           "stream": {"fpr_post": f, "tpr_post": t,
                      "alarm_rate_post": (1 - pi) * f + pi * t}}
    assert _identity(row) == pytest.approx(f)


# ------------------------------------------------------------------ audits

def _row(tag, **kw):
    r = {"tag": tag, "class": "bottle", "seed": 0, "run_hash": tag,
         "config": {"threshold": {"policy": "aci", "aci_update_on": "admitted"},
                    "stream": {"adapt": False, "buffer_protocol": "clean",
                               "admit_explore": 0.0},
                    "drift": {"kind": "combo", "severity": 0.3}},
         "stream": {"fpr_post": 0.05, "n_adaptations": 0},
         "auroc_retention": -0.12}
    for k, v in kw.items():
        if "__" not in k:
            r[k] = v
            continue
        sec, _, key = k.partition("__")
        r["config"][sec][key] = v
    return r


@pytest.fixture
def audit():
    from cgc_ad import audit as A
    A._RESULTS.clear()
    A._DECLARED.clear()
    yield A
    A._RESULTS.clear()
    A._DECLARED.clear()


def test_confound_audit_fails_a_static_arm_that_adapts(audit):
    audit.audit_confounds([_row("r3_static", threshold__policy="static",
                                stream__adapt=True)])
    assert any(s == audit.FAIL for s, _, _ in audit._RESULTS)


def test_confound_audit_flags_drift_the_scorer_cannot_see(audit):
    """brightness @ 0.25 moved clean AUROC by -0.001 in E1."""
    audit.audit_confounds([_row("x", drift__kind="brightness", drift__severity=0.25,
                                auroc_retention=-0.001)])
    assert any(s == audit.WARN and "stationary" in m for s, _, m in audit._RESULTS)


def test_declared_ablation_does_not_fail_the_file(audit):
    bad = _row("r3_naive", stream__buffer_protocol="self_label")
    bad["stream"]["fpr_post"] = 0.74
    audit.audit_degeneracy([bad])
    assert any(s == audit.FAIL for s, _, _ in audit._RESULTS)
    audit._RESULTS.clear()
    audit._DECLARED.add("r3_naive")
    audit.audit_degeneracy([bad])
    assert not any(s == audit.FAIL for s, _, _ in audit._RESULTS)
