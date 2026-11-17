"""Configuration objects for CGC-AD.

Single source of truth for every knob. Configs are plain dataclasses so they
serialise cleanly into the run log, which matters for reproducibility: every
result row carries the exact config that produced it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    root: str = "/kaggle/input/mvtec-ad"
    dataset: str = "mvtec"                 # mvtec | visa
    classes: List[str] = field(default_factory=lambda: ["bottle"])
    image_size: int = 224
    # Three-way split of the *training normals*. Never reuse across roles.
    n_calib: int = 60                      # conformal threshold estimation
    n_val: int = 30                        # model selection / early stopping
    num_workers: int = 2


@dataclass
class DriftConfig:
    """Tier-1 parametric covariate drift applied to the deployment stream."""
    kind: str = "brightness"               # brightness|gamma|blur|noise|contrast|combo|none
    severity: float = 0.6                  # 0..1, mapped per-kind
    onset: int = 300                       # stream step at which drift begins
    ramp: int = 100                        # steps over which severity ramps in (0 = step change)


@dataclass
class SynthConfig:
    """Difficulty-parameterised synthetic anomalies (CutPaste/NSA family).

    `hardness` h in [0, 1]: 0 = trivially visible, 1 = near-invisible.
    All axes are wired so that increasing h monotonically reduces salience;
    whether that translates into monotone *detectability* is an empirical
    question and is exactly what the Phase-1 gate checks.
    """
    n_levels: int = 6                      # discretised hardness ladder
    variants_per_image: int = 2            # cached synthetic variants per (image, level)
    area_easy: float = 0.12                # patch area fraction at h=0
    area_hard: float = 0.008               # ... at h=1
    alpha_easy: float = 1.00               # blend opacity at h=0
    alpha_hard: float = 0.45
    jitter_easy: float = 0.40              # colour delta at h=0
    jitter_hard: float = 0.03
    feather_hard: int = 5                  # edge feathering (px) at h=1
    max_patches: int = 2


@dataclass
class ModelConfig:
    backbone: str = "resnet18"             # resnet18 | wide_resnet50_2
    # Legacy ResNet layer names. Kept so old configs and result rows still
    # load; translated to stage keys internally. Prefer `stages`.
    layers: List[str] = field(default_factory=lambda: ["layer2", "layer3"])
    # Architecture-independent stage keys: s1..s4, coarse to fine. ["s2","s3"]
    # reproduces the original layer2+layer3 setup on any backbone;
    # ["s1","s2","s3"] is the multi-stage / FPN-style arm.
    stages: List[str] = field(default_factory=list)
    pretrained: bool = True
    feat_size: int = 28                    # spatial size features are resampled to
    proj_dim: int = 128                    # random projection; 0 disables
    head_width: int = 128
    topk_frac: float = 0.01                # image score = mean of top-k patch logits


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    # Matched-budget control: every arm gets the same number of gradient steps.
    max_steps: Optional[int] = None
    # cudnn.benchmark picks convolution algorithms by timing, which is not
    # reproducible across runs. The head is tiny; determinism costs little.
    deterministic: bool = True


@dataclass
class CurriculumConfig:
    kind: str = "proportional"             # fixed|random|oracle|proportional|pid|distributional
    h0: float = 0.3
    target_miss: float = 0.15              # gamma
    eta: float = 0.15                      # P gain
    ki: float = 0.02                       # PID
    kd: float = 0.05
    ema: float = 0.5                       # smoothing on the miss-rate estimate
    deadband: float = 0.02
    temperature: float = 0.05              # distributional controller
    rho: float = 0.3                       # distributional controller update rate
    # Floor on every rung's probability in the distributional controller.
    # Without it, the categorical distribution converges to a point mass and
    # reproduces exactly the support collapse that killed the setpoint
    # controller -- the defect this variant was introduced to avoid.
    # Set to 0.0 to reproduce that collapse as an ablation arm.
    min_prob: float = 0.05
    # Controller error signal. This is the F2 ablation axis.
    signal: str = "conformal"              # conformal|raw|synth_percentile|zscore
    raw_threshold: float = 0.0


@dataclass
class ThresholdConfig:
    policy: str = "aci"                    # static|window|aci|robust_aci
    alpha: float = 0.05
    window: int = 200                      # rolling buffer size
    gamma_aci: float = 0.01                # ACI step size
    trim_eps: float = 0.05                 # robust_aci: assumed contamination bound
    max_step: Optional[float] = None       # rate limiter on tau movement
    # Which samples drive the ACI error indicator.
    #   "admitted" reproduces the naive coupling of admission and feedback. With
    #     a self-labelled buffer, admitted := not flagged, so the indicator is
    #     IDENTICALLY ZERO and alpha_t ramps to its cap. Kept only as an
    #     ablation arm demonstrating that degeneracy.
    #   "all" updates alpha_t from every observed sample's flag while the
    #     quantile buffer still takes admitted samples only. This is the fix.
    aci_update_on: str = "all"             # all | admitted


@dataclass
class StreamConfig:
    steps: int = 800
    prevalence: float = 0.05               # fraction of anomalies in the stream
    # clean       admit true nominals            (needs full labels)
    # self_label  admit un-flagged samples       (label-free; censored)
    # gated       admit below a buffer percentile (label-free; censored)
    # reviewed    admit un-flagged samples PLUS flagged samples that a human
    #             disposition marked nominal. Needs labels only on flagged
    #             parts, which inspection lines already route to review.
    #             False alarms are precisely the upper tail of the nominal
    #             distribution that self-labelling censors, so re-admitting
    #             them un-censors the buffer.
    buffer_protocol: str = "clean"
    # eps = P(a true anomaly is admitted to the buffer | it passed the protocol
    # gate). Applied to EVERY protocol, so contamination is a property of the
    # labelling process rather than of one branch. Sweeping this is the
    # contamination experiment; sweeping `prevalence` is a different experiment.
    contamination: float = 0.0
    # Probability of admitting a FLAGGED sample anyway. Without this, a
    # self-labelled buffer satisfies buffer subset {x : x <= tau}, so
    # Q_{1-alpha}(buffer) <= tau for every alpha: tau becomes a monotone
    # non-increasing ratchet and can NEVER adapt upward to drift. Exploration
    # restores an (uncensored) upper tail at the cost of admitting some true
    # anomalies. 0.0 reproduces the ratchet as an ablation arm.
    admit_explore: float = 0.0
    # Clustered defects. Alarm-rate control (aci_update_on="all") raises tau
    # when alarms rise, so it desensitises exactly when defects cluster. A
    # burst of elevated prevalence is the test for that failure mode.
    burst_start: int = -1                  # -1 disables
    burst_len: int = 0
    burst_prevalence: float = 0.40
    gate_percentile: float = 0.5           # for buffer_protocol=gated
    fpr_window: int = 100                  # window for the empirical-FPR trace
    adapt: bool = False                    # enable the online adaptation loop
    adapt_trigger: str = "martingale"      # martingale|ema|timer|none
    adapt_lambda: float = 20.0             # Ville threshold on the martingale
    adapt_steps: int = 100                 # head fine-tune steps per adaptation event
    adapt_period: int = 200                # for trigger=timer
    couple: bool = True                    # RQ3: is hardness re-set from live tau?


@dataclass
class CheckpointConfig:
    enabled: bool = True
    dir: str = "/kaggle/working/cgc_ad_runs"      # small artifacts: persisted
    # Feature banks are large and re-derivable. Default them to the scratch
    # disk (~60 GB, wiped between sessions) rather than /kaggle/working (~20 GB).
    # Set banks_dir under /kaggle/working only if you intend to ship them as a
    # notebook output for the next session to consume.
    banks_dir: str = "/kaggle/temp/cgc_ad_banks"
    cache_banks: bool = True
    # Cache the trained head keyed by bank + training config, so every stream
    # arm reuses one bit-identical head. See checkpoint.head_key.
    head_cache: bool = True
    every_steps: int = 250                        # train-loop checkpoint period
    stream_every: int = 250                       # stream-loop checkpoint period
    keep_last: int = 2
    save_best: bool = True
    best_metric: str = "val_auroc"                # higher is better
    min_free_gb: float = 4.0                      # skip bank caching below this
    resume: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_every: int = 50                           # train steps between scalar logs
    stream_log_every: int = 200
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "cgc-ad"
    save_traces: bool = True


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "/kaggle/working/cgc_ad_runs"
    tag: str = "dev"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Stable run id. Enables --resume to skip completed work."""
        d = self.to_dict()
        for k in ("out_dir", "ckpt", "logging", "device"):
            d.pop(k, None)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


_SECTIONS = {
    "data": DataConfig, "drift": DriftConfig, "synth": SynthConfig,
    "model": ModelConfig, "train": TrainConfig, "curriculum": CurriculumConfig,
    "threshold": ThresholdConfig, "stream": StreamConfig,
    "ckpt": CheckpointConfig, "logging": LoggingConfig,
}


def from_dict(d: Dict[str, Any]) -> Config:
    top = {k: v for k, v in d.items() if k not in _SECTIONS}
    cfg = Config(**top)
    for name, klass in _SECTIONS.items():
        if name in d and d[name] is not None:
            setattr(cfg, name, klass(**d[name]))
    return cfg


def load_yaml(path: str) -> Config:
    import yaml
    with open(path) as f:
        return from_dict(yaml.safe_load(f) or {})


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """`--set curriculum.kind=random threshold.alpha=0.1`"""
    for ov in overrides:
        key, _, raw = ov.partition("=")
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        cur = getattr(obj, parts[-1])
        if isinstance(cur, bool):
            val: Any = raw.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, list):
            val = [s for s in raw.split(",") if s]
        elif cur is None:
            val = None if raw in ("none", "None") else float(raw)
        else:
            val = raw
        setattr(obj, parts[-1], val)
    return cfg
