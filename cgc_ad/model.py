"""Frozen backbone + lightweight patch head.

Rationale (review section 3.2): freezing the backbone
  (a) removes the "you just had more capacity / trained longer" confound,
      because every arm has an identical trainable parameter count;
  (b) makes deployment-time adaptation cheap enough to be credible;
  (c) lets us cache features once and reduce the whole study to head training,
      which is the difference between fitting and not fitting in a T4 quota.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .backbones import MultiStageExtractor, spec_for

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FrozenBackbone(MultiStageExtractor):
    """Backwards-compatible facade over `backbones.MultiStageExtractor`.

    The original implementation registered forward hooks on attributes named
    `layer2`/`layer3`, which exist only on ResNets. It now delegates to the
    graph-node extractor so ViT, Swin, ConvNeXt and the efficient families all
    work, while keeping this constructor signature so existing call sites and
    checkpoints are unaffected.

    Legacy `layers=["layer2","layer3"]` is translated to the stage keys
    `["s2","s3"]`. For `wide_resnet50_2` that reproduces the previous feature
    tensor exactly: the same nodes, the same resample-then-concat order, and a
    projection matrix of the same shape, seed and 1/sqrt(raw_dim) scaling.
    Runs from before this change remain directly comparable.
    """

    _LEGACY = {"layer1": "s1", "layer2": "s2", "layer3": "s3", "layer4": "s4"}

    def __init__(self, name: str = "resnet18",
                 layers: Optional[List[str]] = None,
                 feat_size: int = 28, proj_dim: int = 128, seed: int = 0,
                 stages: Optional[List[str]] = None, pretrained: bool = True):
        keys = list(stages) if stages else None
        if keys is None and layers:
            keys = [self._LEGACY.get(l, l) for l in layers]
        super().__init__(backbone=name, stages=keys, feat_size=feat_size,
                         proj_dim=proj_dim, seed=seed, pretrained=pretrained)
        # kept for call sites that read these attributes
        self.layer_names = [self.spec.stages[k] for k in self.stage_keys]


def build_backbone(cfg, seed: int = 0) -> FrozenBackbone:
    """Construct from a Config, validating the input-size constraint first.

    ViT asserts on any input other than 224x224 deep inside torchvision. Doing
    the check here turns that into an actionable message before a class is
    loaded and a feature bank is built.
    """
    bb = FrozenBackbone(cfg.model.backbone, cfg.model.layers, cfg.model.feat_size,
                        cfg.model.proj_dim, seed=seed,
                        stages=(cfg.model.stages or None),
                        pretrained=getattr(cfg.model, "pretrained", True))
    bb.check_input(cfg.data.image_size)
    return bb


class PatchHead(nn.Module):
    """1x1 conv MLP producing a per-patch anomaly logit."""

    def __init__(self, in_dim: int, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)                                  # (B,1,S,S)


def image_score(patch_logits: torch.Tensor, topk_frac: float = 0.01) -> torch.Tensor:
    """Aggregate patch logits to one score per image (mean of top-k patches).

    Aggregating to the image level before conformalising keeps the calibration
    units exchangeable across images; patch scores within an image are not
    exchangeable, so they are used only for localisation metrics.
    """
    B = patch_logits.shape[0]
    flat = patch_logits.reshape(B, -1)
    k = max(1, int(topk_frac * flat.shape[1]))
    top, _ = torch.topk(flat, k, dim=1)
    return top.mean(dim=1)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
