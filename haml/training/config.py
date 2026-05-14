"""Configuration objects for HAML training orchestration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseConfig:
    """Training phase schedule and core loop settings."""
    n_epochs: int = 50
    batch_size: int = 32
    phase1_epochs: int = 15
    phase2_epochs: int = 15
    td_warmup_power: float = 1.0


@dataclass
class StabilityConfig:
    """Inter-level stability controls."""
    level_divergence_threshold: float = 0.15
    divergence_patience: int = 2
    lr_decay_on_divergence: float = 0.5
    min_lr: float = 1e-4
    early_stop_on_divergence: bool = True
    phase3_only: bool = True
    lr_decay_cooldown_epochs: int = 2
    max_lr_decay_events: int = 4


@dataclass
class AdaptiveMuSepConfig:
    """Adaptive mu_sep schedule to react to divergence."""
    enabled: bool = True
    phase3_only: bool = True
    trigger_divergence: float = 0.05
    patience: int = 1
    growth_factor: float = 1.2
    max_value: float = 1.0


@dataclass
class SoftLandingConfig:
    """Soft-landing safeguards near late-training instability."""
    epoch: Optional[int] = None
    trigger_divergence: Optional[float] = None
    lr_factor: float = 0.1
    freeze_mu: bool = True


@dataclass
class CollapseGuardConfig:
    """One-shot strong reaction to abrupt divergence jumps."""
    enabled: bool = False
    start_epoch: int = 18
    delta_div_threshold: float = 0.10
    mu_sep_boost: float = 1.5
    lr_factor: float = 0.3


@dataclass
class LevelRecoveryConfig:
    """Targeted recovery when one level stays near chance in phase 3."""
    enabled: bool = False
    chance_tolerance: float = 0.02
    patience: int = 3
    max_triggers: int = 1
    require_divergence: float = 0.10
    max_train_accuracy_to_trigger: float = 0.65
    mu_sep_boost: float = 1.2
    lr_factor: float = 0.8
    jitter_std: float = 0.05
    td_cooldown_epochs: int = 1
    td_scale_during_cooldown: float = 0.7
    target_level_idx: Optional[int] = None
