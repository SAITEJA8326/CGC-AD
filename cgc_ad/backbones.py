"""Backbone registry and the multi-stage feature extractor.

WHY THIS MODULE EXISTS

`FrozenBackbone` originally registered forward hooks on attributes literally
named `layer2` and `layer3`. Those names exist only on ResNets. Swapping
`--set model.backbone=vit_b_16` does not work and never could; the extractor
has to be rebuilt around `torchvision.models.feature_extraction`, which
addresses layers by graph node name and works for every architecture.

EVERYTHING BELOW WAS VERIFIED EMPIRICALLY on torchvision 0.19.1 before being
written down (see `python -m cgc_ad.backbones --probe`). Four findings drove
the design:

1. **Node names are family-specific.** ResNet uses `layer2`; ConvNeXt, Swin,
   EfficientNet and MobileNet use `features.N`; ViT uses
   `encoder.layers.encoder_layer_N`. There is no common naming scheme.

2. **Swin returns NHWC**, e.g. `(B, 28, 28, 192)`, while every CNN returns
   NCHW. Feeding Swin features to a conv head without permuting silently
   treats 28 spatial positions as 28 channels. This produces no error and
   garbage results.

3. **ViT returns tokens** `(B, 197, 768)` — 196 patches plus a CLS token that
   must be dropped before reshaping to a 14x14 grid. It is also hard-locked to
   224x224 by an assertion inside torchvision, so `data.image_size` cannot be
   changed for ViT arms.

4. **Concatenated channel dimension varies by more than 12x** across
   candidates: 1536 for WideResNet-50 versus 120 for EfficientNet-B0. Under
   the old code path, `proj_dim=128` would be silently skipped whenever
   `proj_dim > raw_dim`, so EfficientNet arms would feed a 120-channel head
   while ResNet arms fed a 128-channel one — different trainable parameter
   counts, and the budget audit would (correctly) fail the comparison. The
   projection is therefore now ALWAYS applied, which fixes the head width and
   the trainable parameter count identically across every backbone. That is
   what makes a backbone ablation a controlled experiment rather than a
   confounded one.

Backwards compatibility: for `wide_resnet50_2` with stages `["layer2",
"layer3"]` and `proj_dim=128`, raw_dim is 1536 > 128, so the projection was
already being applied under the old code with the same shape, seed and
scaling. Features are bit-identical and existing runs remain comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tvm
from torchvision.models.feature_extraction import create_feature_extractor

# Every torchvision backbone below was verified to use these exact statistics,
# so normalisation is not a confound in the cross-backbone comparison.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class BackboneSpec:
    """Verified description of one backbone. Do not edit from memory: rerun
    `python -m cgc_ad.backbones --probe` and paste the output."""
    ctor: str                          # torchvision constructor name
    paradigm: str                      # what this backbone is a representative OF
    layout: str                        # nchw | nhwc | tokens
    stages: Dict[str, str]             # short key -> graph node name
    channels: Dict[str, int]           # short key -> channel count at 224
    grid: Dict[str, int]               # short key -> spatial size at 224
    default_stages: Tuple[str, ...]    # the two-stage default, matching the
                                       # original layer2+layer3 configuration
    params_m: float
    fixed_input: Optional[int] = None  # hard input-size constraint, if any
    note: str = ""


REGISTRY: Dict[str, BackboneSpec] = {

    # ---- local convolution, the incumbent ---------------------------------
    "wide_resnet50_2": BackboneSpec(
        ctor="wide_resnet50_2", paradigm="local-convolution", layout="nchw",
        stages={"s1": "layer1", "s2": "layer2", "s3": "layer3", "s4": "layer4"},
        channels={"s1": 256, "s2": 512, "s3": 1024, "s4": 2048},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=68.9,
        note="incumbent; s2+s3 reproduces the original layer2+layer3 setup"),

    "resnet18": BackboneSpec(
        ctor="resnet18", paradigm="local-convolution", layout="nchw",
        stages={"s1": "layer1", "s2": "layer2", "s3": "layer3", "s4": "layer4"},
        channels={"s1": 64, "s2": 128, "s3": 256, "s4": 512},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=11.7,
        note="cheap arm for smoke tests"),

    # ---- modernised convolution -------------------------------------------
    "convnext_base": BackboneSpec(
        ctor="convnext_base", paradigm="modernised-convolution", layout="nchw",
        stages={"s1": "features.1", "s2": "features.3",
                "s3": "features.5", "s4": "features.7"},
        channels={"s1": 128, "s2": 256, "s3": 512, "s4": 1024},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=88.6,
        note="large kernels + ViT-style macro design, CNN inductive bias"),

    "convnext_tiny": BackboneSpec(
        ctor="convnext_tiny", paradigm="modernised-convolution", layout="nchw",
        stages={"s1": "features.1", "s2": "features.3",
                "s3": "features.5", "s4": "features.7"},
        channels={"s1": 96, "s2": 192, "s3": 384, "s4": 768},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=28.6,
        note="capacity-matched partner to swin_t"),

    # ---- global attention --------------------------------------------------
    "vit_b_16": BackboneSpec(
        ctor="vit_b_16", paradigm="global-attention", layout="tokens",
        stages={"s1": "encoder.layers.encoder_layer_2",
                "s2": "encoder.layers.encoder_layer_5",
                "s3": "encoder.layers.encoder_layer_8",
                "s4": "encoder.layers.encoder_layer_11"},
        channels={"s1": 768, "s2": 768, "s3": 768, "s4": 768},
        grid={"s1": 14, "s2": 14, "s3": 14, "s4": 14},
        default_stages=("s2", "s3"), params_m=86.6, fixed_input=224,
        note="ALL stages share one 14x14 grid: 'multi-stage' here means depth, "
             "not resolution. torchvision has no vit_s_16; ViT-Small needs timm"),

    # ---- hierarchical windowed attention -----------------------------------
    "swin_t": BackboneSpec(
        ctor="swin_t", paradigm="hierarchical-attention", layout="nhwc",
        stages={"s1": "features.1", "s2": "features.3",
                "s3": "features.5", "s4": "features.7"},
        channels={"s1": 96, "s2": 192, "s3": 384, "s4": 768},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=28.3,
        note="NHWC output; attention WITH a spatial pyramid, so it separates "
             "'attention helps' from 'multi-scale helps' in the ViT result"),

    # ---- compute-constrained ------------------------------------------------
    "efficientnet_b0": BackboneSpec(
        ctor="efficientnet_b0", paradigm="efficient-edge", layout="nchw",
        stages={"s1": "features.2", "s2": "features.3",
                "s3": "features.4", "s4": "features.6"},
        channels={"s1": 24, "s2": 40, "s3": 80, "s4": 192},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=5.3,
        note="raw concat dim is only 120 -- the projection MUST be applied "
             "for the head to match other arms"),

    "mobilenet_v3_large": BackboneSpec(
        ctor="mobilenet_v3_large", paradigm="efficient-edge", layout="nchw",
        stages={"s1": "features.3", "s2": "features.6",
                "s3": "features.12", "s4": "features.16"},
        channels={"s1": 24, "s2": 40, "s3": 112, "s4": 960},
        grid={"s1": 56, "s2": 28, "s3": 14, "s4": 7},
        default_stages=("s2", "s3"), params_m=5.5,
        note="second edge arm; s4 jumps to 960 channels"),
}

# The four-paradigm lineup for the ablation. One representative each, so a
# null result spans genuinely different inductive biases rather than four
# variations on one.
PARADIGM_LINEUP = ["wide_resnet50_2", "convnext_base", "vit_b_16", "swin_t"]


def spec_for(name: str) -> BackboneSpec:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown backbone {name!r}. Known: {sorted(REGISTRY)}. "
            f"To add one, run `python -m cgc_ad.backbones --discover {name}` "
            f"and paste the verified node names into REGISTRY -- never guess "
            f"them, the naming schemes are family-specific.")
    return REGISTRY[name]


# --------------------------------------------------------------------------
# Layout normalisation
# --------------------------------------------------------------------------

def to_nchw(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Bring any backbone's stage output to (B, C, H, W).

    The layout is taken from the registry rather than inferred from the shape.
    Shape sniffing (`is the last dim the odd one out?`) is ambiguous whenever
    C happens to equal H, and a wrong guess produces no error -- just a conv
    head reading spatial positions as channels.
    """
    if layout == "nchw":
        if t.dim() != 4:
            raise ValueError(f"expected 4D NCHW, got {tuple(t.shape)}")
        return t
    if layout == "nhwc":
        if t.dim() != 4:
            raise ValueError(f"expected 4D NHWC, got {tuple(t.shape)}")
        return t.permute(0, 3, 1, 2).contiguous()
    if layout == "tokens":
        if t.dim() != 3:
            raise ValueError(f"expected 3D tokens, got {tuple(t.shape)}")
        b, n, c = t.shape
        g = int(round(n ** 0.5))
        if g * g != n:                       # drop the CLS / distillation token
            t = t[:, n - (int(round((n - 1) ** 0.5)) ** 2):, :]
            n = t.shape[1]
            g = int(round(n ** 0.5))
        if g * g != n:
            raise ValueError(f"token count {n} is not a square after removing "
                             f"prefix tokens")
        return t.transpose(1, 2).reshape(b, c, g, g).contiguous()
    raise ValueError(f"unknown layout {layout!r}")


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------

class MultiStageExtractor(nn.Module):
    """Frozen backbone -> selected stages -> common grid -> concat -> project.

    Args
    ----
    backbone   registry key
    stages     stage keys, e.g. ("s2", "s3") for the two-stage default or
               ("s1", "s2", "s3") for the FPN-style multi-stage arm
    feat_size  spatial size every stage is resampled to before concatenation
    proj_dim   output channels after the fixed random projection. ALWAYS
               applied when > 0, including when it exceeds the raw dimension,
               so the head width and trainable parameter count are identical
               across backbones.
    seed       seeds the projection; the same seed reproduces the same features
    """

    def __init__(self, backbone: str = "wide_resnet50_2",
                 stages: Optional[Sequence[str]] = None,
                 feat_size: int = 28, proj_dim: int = 128, seed: int = 0,
                 pretrained: bool = True):
        super().__init__()
        self.spec = spec_for(backbone)
        self.backbone_name = backbone
        self.stage_keys = tuple(stages) if stages else self.spec.default_stages
        unknown = [s for s in self.stage_keys if s not in self.spec.stages]
        if unknown:
            raise KeyError(f"{backbone} has no stage(s) {unknown}; "
                           f"available: {sorted(self.spec.stages)}")
        self.feat_size = feat_size

        net = self._build(backbone, pretrained)
        nodes = {self.spec.stages[k]: k for k in self.stage_keys}
        self.body = create_feature_extractor(net, return_nodes=nodes)
        self.body.eval()
        for p in self.body.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

        self.raw_dim = sum(self.spec.channels[k] for k in self.stage_keys)
        if proj_dim and proj_dim > 0:
            g = torch.Generator().manual_seed(seed)
            P = torch.randn(proj_dim, self.raw_dim, generator=g) / (self.raw_dim ** 0.5)
            self.register_buffer("proj", P)
            self.out_dim = proj_dim
        else:
            self.proj = None
            self.out_dim = self.raw_dim

    @staticmethod
    def _build(name: str, pretrained: bool) -> nn.Module:
        spec = spec_for(name)
        ctor = getattr(tvm, spec.ctor)
        if not pretrained:
            return ctor(weights=None).eval()
        try:
            return ctor(weights="DEFAULT").eval()
        except TypeError:                                # very old torchvision
            return ctor(pretrained=True).eval()

    def check_input(self, size: int) -> None:
        """ViT asserts on any input other than 224. Fail early and clearly
        rather than deep inside torchvision on the first batch."""
        if self.spec.fixed_input and size != self.spec.fixed_input:
            raise ValueError(
                f"{self.backbone_name} is hard-locked to "
                f"{self.spec.fixed_input}x{self.spec.fixed_input} by "
                f"torchvision; got data.image_size={size}. Either set "
                f"image_size={self.spec.fixed_input} for this arm, or use a "
                f"backbone without a fixed input size.")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] -> (B, out_dim, feat_size, feat_size)"""
        self.check_input(x.shape[-1])
        x = (x - self.mean) / self.std
        out = self.body(x)
        feats = []
        for k in self.stage_keys:
            f = to_nchw(out[k], self.spec.layout)
            if f.shape[-1] != self.feat_size:
                f = F.interpolate(f, size=(self.feat_size, self.feat_size),
                                  mode="bilinear", align_corners=False)
            feats.append(f)
        f = torch.cat(feats, dim=1)
        if self.proj is not None:
            f = torch.einsum("pd,bdhw->bphw", self.proj.to(f.dtype), f)
        return f

    def describe(self) -> Dict[str, object]:
        return {
            "backbone": self.backbone_name,
            "paradigm": self.spec.paradigm,
            "layout": self.spec.layout,
            "stages": list(self.stage_keys),
            "nodes": [self.spec.stages[k] for k in self.stage_keys],
            "native_grids": [self.spec.grid[k] for k in self.stage_keys],
            "feat_size": self.feat_size,
            "raw_dim": self.raw_dim,
            "out_dim": self.out_dim,
            "backbone_params_m": self.spec.params_m,
            "fixed_input": self.spec.fixed_input,
            "upsampled_stages": [k for k in self.stage_keys
                                 if self.spec.grid[k] < self.feat_size],
        }

    def warn_upsampling(self) -> Optional[str]:
        """Interpolating a 14x14 stage to 28x28 adds no information.

        For ViT every stage is 14x14, so `feat_size=28` doubles the token grid
        by interpolation. That is not automatically wrong -- it keeps the head
        and the top-k aggregation identical across arms, which is the right
        default for a controlled comparison -- but it must be stated, and the
        `feat_size=14` sensitivity arm must be run.
        """
        grids = [self.spec.grid[k] for k in self.stage_keys]
        if max(grids) >= self.feat_size:
            return None      # at least one stage carries native detail at this
                             # resolution, which is also true of the incumbent
                             # (layer2 is natively 28) -- nothing to flag
        return (f"{self.backbone_name}: EVERY selected stage has a native grid "
                f"of {grids}, all below feat_size={self.feat_size}. The output "
                f"map is entirely interpolated: no stage contributes detail at "
                f"this resolution. Keeping feat_size fixed holds the head and "
                f"the top-k aggregation identical across arms, which is the "
                f"right default for a controlled comparison, but it must be "
                f"stated in the paper and the feat_size={max(grids)} "
                f"sensitivity arm must be run.")


# --------------------------------------------------------------------------
# Verification CLI -- run this BEFORE integrating anything
# --------------------------------------------------------------------------

def probe(names: Optional[Sequence[str]] = None, image_size: int = 224,
          feat_size: int = 28, proj_dim: int = 128,
          stages: Optional[Sequence[str]] = None) -> List[Dict]:
    """Instantiate each backbone, run one batch, and check the registry
    against reality. Any mismatch is a registry bug, not a runtime bug."""
    out = []
    for name in (names or list(REGISTRY)):
        row: Dict[str, object] = {"backbone": name}
        try:
            ex = MultiStageExtractor(name, stages, feat_size, proj_dim,
                                     pretrained=False)
            size = ex.spec.fixed_input or image_size
            x = torch.zeros(2, 3, size, size)
            with torch.no_grad():
                raw = ex.body(x)
            row["raw_shapes"] = {k: tuple(v.shape) for k, v in raw.items()}
            nchw = {k: tuple(to_nchw(v, ex.spec.layout).shape)
                    for k, v in raw.items()}
            row["nchw_shapes"] = nchw
            with torch.no_grad():
                y = ex(x)
            row.update(ex.describe())
            row["output_shape"] = tuple(y.shape)
            # registry cross-check
            bad = []
            for k in ex.stage_keys:
                c_reg, g_reg = ex.spec.channels[k], ex.spec.grid[k]
                c_act, g_act = nchw[k][1], nchw[k][2]
                if (c_reg, g_reg) != (c_act, g_act):
                    bad.append(f"{k}: registry says C={c_reg},G={g_reg}; "
                               f"actual C={c_act},G={g_act}")
            row["registry_ok"] = not bad
            row["registry_errors"] = bad
            row["warning"] = ex.warn_upsampling()
            row["ok"] = True
        except Exception as e:
            row["ok"] = False
            row["error"] = f"{type(e).__name__}: {e}"
        out.append(row)
    return out


def discover(name: str, image_size: int = 224, top: int = 40) -> None:
    """Print candidate graph nodes and their shapes for an unregistered model.

    Use this to add a backbone. Copy the verified node names and shapes into
    REGISTRY; do not write them from memory.
    """
    from torchvision.models.feature_extraction import get_graph_node_names
    net = getattr(tvm, name)(weights=None).eval()
    train_nodes, _ = get_graph_node_names(net)
    print(f"{name}: {len(train_nodes)} graph nodes, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params")
    print("top-level children:", [k for k, _ in net.named_children()])
    cands = [n for n in train_nodes
             if n.count(".") <= 2 and not n.startswith(("x", "getattr"))]
    x = torch.zeros(1, 3, image_size, image_size)
    shown = 0
    for n in cands:
        try:
            fx = create_feature_extractor(net, {n: n})
            with torch.no_grad():
                s = tuple(fx(x)[n].shape)
            if len(s) in (3, 4):
                print(f"  {n:46s} {s}")
                shown += 1
                if shown >= top:
                    break
        except Exception:
            continue


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser("cgc-ad-backbones")
    ap.add_argument("--probe", action="store_true",
                    help="verify every registered backbone against reality")
    ap.add_argument("--discover", default=None,
                    help="print graph nodes and shapes for a new model")
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--feat-size", type=int, default=28)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.discover:
        discover(args.discover)
        return 0

    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}\n")
    rows = probe(args.backbones, feat_size=args.feat_size,
                 proj_dim=args.proj_dim, stages=args.stages)
    bad = 0
    for r in rows:
        if not r.get("ok"):
            print(f"[FAIL] {r['backbone']:20s} {r['error']}")
            bad += 1
            continue
        flag = "ok  " if r["registry_ok"] else "MISMATCH"
        print(f"[{flag}] {r['backbone']:20s} {r['paradigm']:24s} "
              f"layout={r['layout']:6s} stages={r['stages']} "
              f"raw_dim={r['raw_dim']:5d} -> out={r['output_shape']} "
              f"({r['backbone_params_m']:.1f}M frozen)")
        for k, v in r["raw_shapes"].items():
            print(f"           {k}: raw {v} -> nchw {r['nchw_shapes'][k]}")
        for e in r["registry_errors"]:
            print(f"           REGISTRY ERROR: {e}")
            bad += 1
        if r.get("warning"):
            print(f"           note: {r['warning']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    print(f"\n{len(rows) - bad}/{len(rows)} backbones verified against the registry")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
