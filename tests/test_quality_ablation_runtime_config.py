import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.quality_ablation_runtime import build_model_overrides


def _args(dataset="fashion_mnist", device="cpu"):
    return argparse.Namespace(dataset=dataset, device=device)


def test_build_model_overrides_applies_fashion_cpu_reference_defaults():
    resolved = build_model_overrides(_args("fashion_mnist", "cpu"), {})
    assert resolved["sigma_init_mode"] == "sqrt_d_std"
    assert resolved["learn_projections"] is True


def test_build_model_overrides_variant_override_has_priority():
    resolved = build_model_overrides(
        _args("fashion_mnist", "cpu"),
        {"sigma_init_mode": "median_pairwise", "learn_projections": False},
    )
    assert resolved["sigma_init_mode"] == "median_pairwise"
    assert resolved["learn_projections"] is False


def test_build_model_overrides_no_forcing_outside_fashion_cpu():
    resolved = build_model_overrides(_args("make_moons", "cpu"), {})
    assert resolved == {}
