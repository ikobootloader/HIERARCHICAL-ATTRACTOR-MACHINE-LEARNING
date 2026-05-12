"""
Recherche d'hyperparametres SA-nODE-reimpl (seed fixe).

Optimise uniquement des hyperparametres dynamiques:
- a (amplitude attracteurs)
- dt (pas d'integration)
- gamma (amortissement)

Architecture conservee:
- espace d'etat plat
- attracteurs plantes
- pas de hierarchie
"""

import json
import random
import sys
from pathlib import Path
from itertools import product

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haml.baselines import SANodeReimpl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_config(a, dt, gamma, seed=42):
    noise_levels = [0.10, 0.20, 0.30, 0.40]
    n_samples = 3000
    per_noise = []

    for noise in noise_levels:
        set_seed(seed)
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SANodeReimpl(
            a=a,
            beta=0.8,
            gamma=gamma,
            dt=dt,
            n_steps=120,
            lr=0.02,
            n_epochs=120,
            lambda_kernel=2.0,
            lambda_l2=1e-3,
            seed=seed,
            device="cpu",
        )
        model.fit(X_train, y_train)
        acc = float(model.score(X_test, y_test))
        per_noise.append({"noise": noise, "accuracy": acc})

    accs = [x["accuracy"] for x in per_noise]
    return {
        "params": {"a": a, "dt": dt, "gamma": gamma},
        "per_noise": per_noise,
        "mean_accuracy": float(np.mean(accs)),
        "accuracy_noise_040": float(per_noise[-1]["accuracy"]),
    }


def main():
    seed = 42
    grid_a = [0.8, 1.0, 1.2]
    grid_dt = [0.03, 0.05, 0.08]
    grid_gamma = [0.0, 0.05, 0.10]

    results = []
    for a, dt, gamma in product(grid_a, grid_dt, grid_gamma):
        results.append(evaluate_config(a=a, dt=dt, gamma=gamma, seed=seed))

    # Selection principale: meilleure moyenne multi-bruit
    best = max(results, key=lambda r: r["mean_accuracy"])

    # Selection secondaire: meilleure perf en bruit fort
    best_040 = max(results, key=lambda r: r["accuracy_noise_040"])

    out = {
        "seed": seed,
        "search_space": {"a": grid_a, "dt": grid_dt, "gamma": grid_gamma},
        "objective_primary": "maximize mean accuracy over noise levels [0.10, 0.20, 0.30, 0.40]",
        "objective_secondary": "maximize accuracy at noise=0.40",
        "best_primary": best,
        "best_noise_040": best_040,
        "all_results": results,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

