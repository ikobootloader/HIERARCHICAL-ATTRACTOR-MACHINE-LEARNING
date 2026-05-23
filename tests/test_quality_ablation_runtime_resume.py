import argparse
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.quality_ablation_runtime import _base_report, _load_resume_report


def _args(out_json, resume=True):
    return argparse.Namespace(
        ablation="b3",
        dataset="fashion_mnist",
        seeds=[42, 43],
        n_train=5000,
        n_test=1000,
        n_epochs=10,
        phase1_epochs=3,
        phase2_epochs=3,
        training_preset="ultra_fast_train_cpu",
        batch_size=128,
        device="cpu",
        out_json=str(out_json),
        resume=resume,
    )


def test_resume_loads_matching_file(tmp_path):
    out = tmp_path / "resume.json"
    args = _args(out, resume=True)
    expected_variants = [("proj_fixed", {}), ("proj_learned", {})]
    payload = _base_report(args)
    payload["variants"] = {
        "proj_fixed": {"runs": [{"seed": 42, "train_time_sec": 1.0, "test_accuracy": 0.8, "final_train_accuracy": 0.9}]}
    }
    out.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_resume_report(args, expected_variants)
    assert loaded["ablation"] == "b3"
    assert "proj_fixed" in loaded["variants"]
    assert "proj_learned" in loaded["variants"]


def test_resume_rejects_mismatched_config(tmp_path):
    out = tmp_path / "resume_bad.json"
    args = _args(out, resume=True)
    expected_variants = [("proj_fixed", {}), ("proj_learned", {})]
    payload = _base_report(args)
    payload["config"]["n_epochs"] = 9  # mismatch
    out.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        _load_resume_report(args, expected_variants)
