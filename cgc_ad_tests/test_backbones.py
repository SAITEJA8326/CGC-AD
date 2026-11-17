"""Backbone registry, layout normalisation, and cross-backbone controls.

WHY THIS FILE EXISTS

The old extractor hooked attributes named `layer2`/`layer3`, which exist only
on ResNets, so `--set model.backbone=vit_b_16` could never have worked. The
replacement addresses layers by graph node name and must therefore handle three
different output conventions. Two of the failure modes are silent:

  * Swin emits NHWC. Feeding that to a conv head without permuting reads 28
    spatial positions as 28 channels. No exception, no NaN, just noise.
  * ViT emits tokens with a CLS prefix. Reshaping 197 tokens without dropping
    the CLS either throws or, with the wrong arithmetic, misaligns every patch.

And one is a confound rather than a bug: EfficientNet-B0's concatenated stages
are 120 channels, below the projection width, so the old code skipped the
projection and gave that arm a differently-shaped head. A backbone ablation
where the head width varies by backbone is not a controlled experiment.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="backbone tests need torch")

from cgc_ad.backbones import (PARADIGM_LINEUP, REGISTRY, MultiStageExtractor,
                              probe, spec_for, to_nchw)
from cgc_ad.config import Config
from cgc_ad.model import FrozenBackbone, PatchHead, build_backbone, count_trainable

ALL = sorted(REGISTRY)


# ------------------------------------------------------------------ registry

@pytest.mark.parametrize("name", ALL)
def test_registry_matches_reality(name):
    """The registry records node names, channel counts and grid sizes that were
    read off a real forward pass. If torchvision renames a node in a future
    release, this fails loudly instead of the extractor silently pulling the
    wrong layer."""
    r = probe([name])[0]
    assert r["ok"], r.get("error")
    assert r["registry_ok"], r["registry_errors"]


def test_lineup_covers_four_distinct_paradigms():
    """A null result across four ResNet variants is not evidence about
    architecture. The lineup has to span genuinely different inductive biases
    for 'the backbone is not the bottleneck' to mean anything."""
    paradigms = {spec_for(n).paradigm for n in PARADIGM_LINEUP}
    assert len(paradigms) == len(PARADIGM_LINEUP), paradigms
    assert {"local-convolution", "global-attention"} <= paradigms


def test_unknown_backbone_gives_an_actionable_error():
    with pytest.raises(KeyError, match="discover"):
        spec_for("vit_s_16")            # not in torchvision; timm only


# ------------------------------------------------------- layout normalisation

def test_nhwc_is_permuted_not_reinterpreted():
    """Swin's (B,H,W,C) must become (B,C,H,W) by permutation. A reshape would
    preserve the element count and silently scramble the feature map."""
    x = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 4, 4, 3)
    y = to_nchw(x, "nhwc")
    assert y.shape == (2, 3, 4, 4)
    assert torch.equal(y[0, :, 1, 2], x[0, 1, 2, :])


def test_tokens_drop_the_cls_token_and_form_a_square_grid():
    """197 tokens = 1 CLS + 196 patches. Keeping the CLS makes 197 non-square
    and misaligns the grid."""
    x = torch.randn(2, 197, 768)
    y = to_nchw(x, "tokens")
    assert y.shape == (2, 768, 14, 14)
    assert torch.equal(y[0, :, 0, 0], x[0, 1, :])       # first PATCH, not CLS
    y2 = to_nchw(torch.randn(2, 196, 768), "tokens")    # already square
    assert y2.shape == (2, 768, 14, 14)


def test_layout_is_declared_not_sniffed():
    """Inferring layout from shape is ambiguous when C == H (e.g. 28 channels
    at 28x28). The registry declares it, so this case is unambiguous."""
    x = torch.randn(2, 28, 28, 28)
    assert to_nchw(x, "nchw").shape == (2, 28, 28, 28)
    assert to_nchw(x, "nhwc").shape == (2, 28, 28, 28)
    assert not torch.equal(to_nchw(x, "nchw"), to_nchw(x, "nhwc"))


# ------------------------------------------------------ cross-backbone control

@pytest.mark.parametrize("name", PARADIGM_LINEUP)
def test_every_backbone_yields_the_same_output_shape(name):
    """The whole point of the projection: identical (C,H,W) out of every
    backbone, so the head is byte-for-byte the same module in every arm."""
    ex = MultiStageExtractor(name, feat_size=28, proj_dim=128, pretrained=False)
    size = ex.spec.fixed_input or 224
    with torch.no_grad():
        y = ex(torch.zeros(2, 3, size, size))
    assert y.shape == (2, 128, 28, 28)


def test_head_parameter_count_is_identical_across_backbones():
    """THE CONTROL THAT MAKES THE ABLATION VALID. EfficientNet-B0's raw dim is
    120 < proj_dim=128; under the old code the projection was skipped and that
    arm got a 120-channel head. Different trainable parameter counts across
    arms means the budget audit fails and the comparison is confounded."""
    counts = {}
    for name in ["wide_resnet50_2", "vit_b_16", "swin_t", "efficientnet_b0"]:
        ex = MultiStageExtractor(name, proj_dim=128, pretrained=False)
        counts[name] = count_trainable(PatchHead(ex.out_dim, 128))
    assert len(set(counts.values())) == 1, counts


def test_projection_is_applied_even_when_it_expands():
    ex = MultiStageExtractor("efficientnet_b0", proj_dim=128, pretrained=False)
    assert ex.raw_dim == 120 and ex.out_dim == 128
    assert ex.proj is not None and ex.proj.shape == (128, 120)


def test_projection_is_deterministic_given_the_seed():
    a = MultiStageExtractor("resnet18", proj_dim=128, seed=7, pretrained=False)
    b = MultiStageExtractor("resnet18", proj_dim=128, seed=7, pretrained=False)
    c = MultiStageExtractor("resnet18", proj_dim=128, seed=8, pretrained=False)
    assert torch.equal(a.proj, b.proj) and not torch.equal(a.proj, c.proj)


# ------------------------------------------------------------- multi-stage arm

@pytest.mark.parametrize("name", PARADIGM_LINEUP)
def test_multi_stage_adds_channels_without_changing_the_interface(name):
    """The FPN-style arm concatenates s1+s2+s3. The raw dimension grows, the
    output shape does not, so multi-stage vs two-stage is a controlled
    comparison rather than a capacity change in the head."""
    two = MultiStageExtractor(name, ["s2", "s3"], pretrained=False)
    three = MultiStageExtractor(name, ["s1", "s2", "s3"], pretrained=False)
    assert three.raw_dim > two.raw_dim
    assert three.out_dim == two.out_dim
    size = two.spec.fixed_input or 224
    with torch.no_grad():
        assert three(torch.zeros(1, 3, size, size)).shape == \
               two(torch.zeros(1, 3, size, size)).shape


def test_bad_stage_key_is_rejected():
    with pytest.raises(KeyError):
        MultiStageExtractor("vit_b_16", ["s9"], pretrained=False)


# ------------------------------------------------------------ input constraint

def test_vit_input_size_constraint_fails_early_and_clearly():
    """torchvision asserts 'Wrong image height!' from deep inside the model on
    the first batch, after a class has been loaded and a feature bank started.
    Checking at construction turns that into an actionable message."""
    cfg = Config()
    cfg.model.backbone = "vit_b_16"
    cfg.model.pretrained = False          # no network in CI
    cfg.data.image_size = 256
    with pytest.raises(ValueError, match="hard-locked"):
        build_backbone(cfg)
    cfg.data.image_size = 224
    build_backbone(cfg)                      # no raise


def test_only_vit_declares_a_fixed_input():
    fixed = {n for n in ALL if spec_for(n).fixed_input}
    assert fixed == {"vit_b_16"}


# --------------------------------------------------------- upsampling honesty

def test_vit_is_flagged_as_fully_interpolated_and_resnet_is_not():
    """For WideResNet, s2 is natively 28x28, so the 28x28 map carries real
    detail. For ViT every stage is 14x14 and the map is entirely interpolated.
    That difference has to be reported, not discovered by a reviewer."""
    vit = MultiStageExtractor("vit_b_16", feat_size=28, pretrained=False)
    wrn = MultiStageExtractor("wide_resnet50_2", feat_size=28, pretrained=False)
    assert vit.warn_upsampling() is not None
    assert wrn.warn_upsampling() is None
    assert MultiStageExtractor("vit_b_16", feat_size=14,
                               pretrained=False).warn_upsampling() is None


# -------------------------------------------------------------- compatibility

def test_legacy_layer_names_still_work():
    """Existing configs and result rows carry layers=['layer2','layer3'].
    They must keep loading."""
    bb = FrozenBackbone("wide_resnet50_2", ["layer2", "layer3"], 28, 128,
                        pretrained=False)
    assert bb.stage_keys == ("s2", "s3")
    assert bb.layer_names == ["layer2", "layer3"]
    assert bb.raw_dim == 1536


def test_resnet_path_is_feature_identical_to_the_hook_implementation():
    """REGRESSION GUARD. The refactor must not move the incumbent's numbers, or
    every run completed before it becomes incomparable with every run after."""
    import torch.nn.functional as Fn
    import torchvision
    name = "resnet18"
    new = FrozenBackbone(name, ["layer2", "layer3"], 28, 128, seed=0,
                         pretrained=False).eval()
    net = getattr(torchvision.models, name)(weights=None).eval()
    new.body.load_state_dict(net.state_dict(), strict=False)
    feats = {}
    for ln in ("layer2", "layer3"):
        getattr(net, ln).register_forward_hook(
            lambda m, i, o, ln=ln: feats.__setitem__(ln, o))
    x = torch.rand(2, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        net((x - mean) / std)
        ref = torch.cat([Fn.interpolate(feats[l], size=(28, 28), mode="bilinear",
                                        align_corners=False)
                         for l in ("layer2", "layer3")], dim=1)
        ref = torch.einsum("pd,bdhw->bphw", new.proj, ref)
        got = new(x)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max().item()


def test_bank_cache_key_separates_backbones_and_stages():
    """Feature banks are cached on disk. If the key ignored the backbone, a
    ViT run would silently load ResNet features and the ablation would compare
    a backbone against itself."""
    from cgc_ad.checkpoint import banks_key
    base = Config()
    keys = set()
    for bb, st in [("wide_resnet50_2", []), ("vit_b_16", []),
                   ("wide_resnet50_2", ["s1", "s2", "s3"]),
                   ("swin_t", ["s2", "s3"])]:
        c = Config()
        c.model.backbone, c.model.stages = bb, st
        keys.add(banks_key(c, "bottle"))
    assert len(keys) == 4
