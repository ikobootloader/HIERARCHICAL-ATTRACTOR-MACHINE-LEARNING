"""
Benchmark RK4 vs Adjoint on the same HAML configuration.
"""

import json
import random
import time
import tracemalloc
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haml import HAML


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_once(method, seed=42):
    set_seed(seed)
    X, y = make_moons(n_samples=1800, noise=0.25, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = HAML(
        n_levels=3,
        n_attractors_per_class=5,
        alpha_bu=0.5,
        alpha_td=1.5,
        integrator_method=method,
        max_steps=35,
        dt=0.1,
        lr=0.01,
        n_epochs=5,
        batch_size=64,
        train_on_fit=False,
        device="cpu",
    )

    tracemalloc.start()
    t0 = time.time()
    model.fit(X_train, y_train)
    model.train_model(X_train, y_train)
    acc = float(model.score(X_test, y_test))
    elapsed = time.time() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": method,
        "accuracy": acc,
        "elapsed_s": elapsed,
        "peak_tracemalloc_mb": peak / (1024 * 1024),
    }


def main():
    seed = 42
    methods = ["rk4", "adjoint"]
    results = [run_once(m, seed=seed) for m in methods]

    rk4 = next(r for r in results if r["method"] == "rk4")
    adj = next(r for r in results if r["method"] == "adjoint")
    summary = {
        "seed": seed,
        "config": {
            "dataset": "make_moons",
            "n_samples": 1800,
            "noise": 0.25,
            "test_size": 0.25,
            "n_epochs": 5,
            "batch_size": 64,
            "max_steps": 35,
            "dt": 0.1,
            "alpha_bu": 0.5,
            "alpha_td": 1.5,
        },
        "results": results,
        "relative": {
            "adjoint_vs_rk4_time_ratio": adj["elapsed_s"] / rk4["elapsed_s"] if rk4["elapsed_s"] > 0 else None,
            "adjoint_minus_rk4_accuracy_points": (adj["accuracy"] - rk4["accuracy"]) * 100.0,
            "adjoint_minus_rk4_peak_tracemalloc_mb": adj["peak_tracemalloc_mb"] - rk4["peak_tracemalloc_mb"],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

