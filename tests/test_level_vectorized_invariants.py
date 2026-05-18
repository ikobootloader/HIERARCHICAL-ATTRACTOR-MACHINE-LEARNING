import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml.dynamics.level import Level
from haml.dynamics.level_vectorized import LevelVectorized


def _build_levels(seed=123):
    torch.manual_seed(seed)
    n_classes = 3
    n_attr = 2
    dim = 8
    level_ref = Level(
        level_idx=0,
        dim=dim,
        n_classes=n_classes,
        n_attractors_per_class=n_attr,
        lambda_repulsion=0.5,
        rho_sigma_ratio=2.0,
    )
    level_vec = LevelVectorized(
        level_idx=0,
        dim=dim,
        n_classes=n_classes,
        n_attractors_per_class=n_attr,
        lambda_repulsion=0.5,
        rho_sigma_ratio=2.0,
        repulsion_mode="global",
    )

    X = torch.randn(120, dim)
    y = torch.randint(0, n_classes, (120,))
    # Guarantee all classes appear.
    y[:n_classes] = torch.arange(0, n_classes)
    level_ref.initialize_attractors(X, y)

    with torch.no_grad():
        for c in range(n_classes):
            for m, attractor in enumerate(level_ref.attractors[str(c)]):
                idx = c * n_attr + m
                level_vec.positions[idx].copy_(attractor.position)
                level_vec.log_sigma[idx].copy_(attractor.log_sigma)
                level_vec.log_rho[idx].copy_(attractor.log_rho)
                level_vec.log_weight[idx].copy_(attractor.log_weight)
    return level_ref, level_vec


def test_vectorized_intra_level_force_matches_reference():
    level_ref, level_vec = _build_levels()
    x = torch.randn(16, 8)
    ref_force = level_ref.intra_level_force(x)
    vec_force = level_vec.intra_level_force(x)
    assert torch.allclose(ref_force, vec_force, atol=1e-5, rtol=1e-5)


def test_vectorized_predict_scores_match_reference():
    level_ref, level_vec = _build_levels()
    x = torch.randn(16, 8)
    ref_scores = level_ref.predict_class_scores(x)
    vec_scores = level_vec.predict_class_scores(x)
    assert torch.allclose(ref_scores, vec_scores, atol=1e-5, rtol=1e-5)


def test_vectorized_separation_energy_matches_reference():
    level_ref, level_vec = _build_levels()
    ref_val = level_ref.separation_energy()
    vec_val = level_vec.separation_energy()
    assert torch.allclose(ref_val, vec_val, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fn_name", ["intra_level_force", "predict_class_scores"])
def test_vectorized_backward_matches_reference(fn_name):
    level_ref, level_vec = _build_levels()
    x_ref = torch.randn(12, 8, requires_grad=True)
    x_vec = x_ref.clone().detach().requires_grad_(True)

    ref_out = getattr(level_ref, fn_name)(x_ref).sum()
    vec_out = getattr(level_vec, fn_name)(x_vec).sum()
    ref_out.backward()
    vec_out.backward()

    assert torch.allclose(x_ref.grad, x_vec.grad, atol=1e-5, rtol=1e-5)
