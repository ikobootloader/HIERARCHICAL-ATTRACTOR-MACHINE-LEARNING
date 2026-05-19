import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Priorise le package haml de github_app (version active du projet).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml.dynamics.level import Level
from haml.model.haml import HAML
from haml.training import (
    AdaptiveMuSepConfig,
    CollapseGuardConfig,
    HAMLTrainer,
    PhaseConfig,
    SoftLandingConfig,
    StabilityConfig,
)


class _DummyLevel(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self._scores = scores

    def predict_class_scores(self, _state):
        return self._scores


class _DummyCoupling:
    def __init__(self):
        self.alpha_bu = torch.tensor(0.5)
        self.alpha_td = torch.tensor(1.0)


class _DummyModel:
    def __init__(self):
        self.coupling = _DummyCoupling()
        self.levels = []


class _DummyOptimizer:
    pass


class _DummyLoss:
    pass


def test_level_weighting_modes():
    model = HAML(level_score_weighting="exponential")
    assert model._level_weight(0) == 1.0
    assert model._level_weight(1) == 2.0
    assert model._level_weight(2) == 4.0

    model_uniform = HAML(level_score_weighting="uniform")
    assert model_uniform._level_weight(0) == 1.0
    assert model_uniform._level_weight(3) == 1.0

    model_invalid = HAML(level_score_weighting="invalid")
    with pytest.raises(ValueError):
        model_invalid._level_weight(0)


def test_hierarchical_score_aggregation_is_centralized():
    model = HAML(level_score_weighting="exponential")
    model.n_classes_ = 2
    model.device = "cpu"
    model.levels = nn.ModuleList(
        [
            _DummyLevel(torch.tensor([[1.0, 0.0]])),
            _DummyLevel(torch.tensor([[0.0, 2.0]])),
        ]
    )
    final_states = [torch.zeros(1, 1), torch.zeros(1, 1)]

    scores = model._compute_hierarchical_scores(final_states)
    expected = torch.tensor([[1.0, 4.0]])  # 1*level0 + 2*level1
    assert torch.allclose(scores, expected)


def test_level_rho_sigma_ratio_is_applied_at_init():
    level = Level(
        level_idx=0,
        dim=2,
        n_classes=2,
        n_attractors_per_class=1,
        rho_sigma_ratio=3.0,
    )
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [1.0, 1.0],
            [1.1, 0.9],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    level.initialize_attractors(X, y)
    for class_idx in range(level.n_classes):
        attractor = level.attractors[str(class_idx)][0]
        ratio = (attractor.rho / attractor.sigma).item()
        assert np.isclose(ratio, 3.0, atol=1e-5)
        assert attractor.rho.item() > attractor.sigma.item()


def test_level_initialize_raises_on_missing_class_samples():
    level = Level(level_idx=0, dim=2, n_classes=2, n_attractors_per_class=1)
    X = torch.tensor([[0.0, 0.0], [0.2, 0.1]], dtype=torch.float32)
    y = torch.tensor([0, 0], dtype=torch.long)  # classe 1 absente

    with pytest.raises(ValueError, match="class 1 has no samples"):
        level.initialize_attractors(X, y)


def test_trainer_accepts_grouped_configs():
    trainer = HAMLTrainer(
        model=_DummyModel(),
        optimizer=_DummyOptimizer(),
        loss_fn=_DummyLoss(),
        phase_config=PhaseConfig(n_epochs=10, batch_size=16, phase1_epochs=3, phase2_epochs=3),
        stability_config=StabilityConfig(level_divergence_threshold=0.2, divergence_patience=3),
        adaptive_mu_sep_config=AdaptiveMuSepConfig(enabled=False),
        soft_landing_config=SoftLandingConfig(lr_factor=0.2),
        collapse_guard_config=CollapseGuardConfig(enabled=True),
        verbose=False,
    )

    assert trainer.phase_config.n_epochs == 10
    assert trainer.phase_config.batch_size == 16
    assert trainer.stability_config.level_divergence_threshold == 0.2
    assert trainer.adaptive_mu_sep_config.enabled is False
    assert trainer.soft_landing_config.lr_factor == 0.2
    assert trainer.collapse_guard_config.enabled is True


def test_haml_exposes_phase_runtime_knobs_in_get_params():
    model = HAML(
        phase1_max_steps=20,
        phase2_max_steps=40,
        phase3_max_steps=100,
        phase1_integrator_method="euler",
        phase2_integrator_method="euler",
        phase3_integrator_method="rk4",
        diagnostics_every_epochs=2,
        diagnostics_subset_size=512,
    )
    params = model.get_params()
    assert params["phase1_max_steps"] == 20
    assert params["phase2_max_steps"] == 40
    assert params["phase3_max_steps"] == 100
    assert params["phase1_integrator_method"] == "euler"
    assert params["phase2_integrator_method"] == "euler"
    assert params["phase3_integrator_method"] == "rk4"
    assert params["diagnostics_every_epochs"] == 2
    assert params["diagnostics_subset_size"] == 512


def test_haml_fast_train_cpu_preset_applies_defaults():
    model = HAML(training_preset="fast_train_cpu")
    params = model.get_params()
    assert params["training_preset"] == "fast_train_cpu"
    assert params["phase1_max_steps"] == 20
    assert params["phase2_max_steps"] == 40
    assert params["phase3_max_steps"] == 100
    assert params["phase1_integrator_method"] == "euler"
    assert params["phase2_integrator_method"] == "euler"
    assert params["phase3_integrator_method"] == "rk4"


def test_haml_fast_train_cpu_preset_respects_explicit_overrides():
    model = HAML(
        training_preset="fast_train_cpu",
        phase1_max_steps=30,
        phase1_integrator_method="rk4",
    )
    params = model.get_params()
    assert params["phase1_max_steps"] == 30
    assert params["phase1_integrator_method"] == "rk4"
    assert params["phase2_max_steps"] == 40
    assert params["phase2_integrator_method"] == "euler"
