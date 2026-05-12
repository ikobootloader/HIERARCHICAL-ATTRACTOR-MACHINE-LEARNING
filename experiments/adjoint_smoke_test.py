"""
Smoke test for HAML adjoint integrator path.

Runs a small make_moons training using integrator_method='adjoint' and
prints accuracy + elapsed time.
"""

import time
import random
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Force local github_app package resolution before any installed 'haml'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haml import HAML


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed = 42
    set_seed(seed)
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=seed)
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
        integrator_method="adjoint",
        max_steps=30,
        dt=0.1,
        lr=0.01,
        n_epochs=3,
        batch_size=64,
        train_on_fit=False,
        device="cpu",
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    model.train_model(X_train, y_train)
    acc = float(model.score(X_test, y_test))
    elapsed = time.time() - t0

    print(
        f"adjoint_smoke_test: acc={acc:.4f}, elapsed_s={elapsed:.2f}, method={model.integrator_method}"
    )


if __name__ == "__main__":
    main()
