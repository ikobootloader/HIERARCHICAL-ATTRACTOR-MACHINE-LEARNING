import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.quality_ablation_runtime import build_variants


def test_build_variants_b5_level_weighting():
    variants = build_variants("b5")
    names = [name for name, _ in variants]
    assert "level_weight_exponential" in names
    assert "level_weight_uniform" in names
    assert "level_weight_learned_softmax" in names
