"""
Ablation MNIST: niveaux indépendants vs couplage bidirectionnel.

Compare deux configurations strictement identiques, sauf alpha_bu/alpha_td:
- Independent: alpha_bu=0.0, alpha_td=0.0
- Coupled: alpha_bu=1.0, alpha_td=1.0
"""

import json
import os
import time
import random
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from haml import HAML


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist_subset(n_train=1500, n_test=500, seed=42):
    cache_dir = os.path.join(os.getcwd(), ".sklearn_data")
    os.makedirs(cache_dir, exist_ok=True)

    mnist = fetch_openml("mnist_784", version=1, parser="auto", data_home=cache_dir)
    X = mnist.data.to_numpy()
    y = mnist.target.to_numpy().astype(int)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n_train + n_test, replace=False)
    X = X[idx]
    y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def run_config(name, alpha_bu, alpha_td, X_train, y_train, X_test, y_test, seed=42):
    set_seed(seed)

    model = HAML(
        n_levels=3,
        n_attractors_per_class=2,
        alpha_bu=alpha_bu,
        alpha_td=alpha_td,
        max_steps=20,
        lr=0.01,
        n_epochs=5,
        batch_size=128,
        train_on_fit=False,
        device="cpu",
    )

    model.fit(X_train, y_train)

    start = time.time()
    history = model.train_model(X_train, y_train)
    train_time = time.time() - start

    test_acc = model.score(X_test, y_test)

    return {
        "name": name,
        "alpha_bu": alpha_bu,
        "alpha_td": alpha_td,
        "train_accuracy_by_epoch": history["accuracy"],
        "test_accuracy": float(test_acc),
        "train_time_sec": float(train_time),
    }


def main():
    seed = 42
    X_train, X_test, y_train, y_test = load_mnist_subset(
        n_train=1500, n_test=500, seed=seed
    )

    independent = run_config(
        name="independent",
        alpha_bu=0.0,
        alpha_td=0.0,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )

    coupled = run_config(
        name="coupled",
        alpha_bu=1.0,
        alpha_td=1.0,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )

    summary = {
        "dataset": {"train": 1500, "test": 500, "seed": seed},
        "independent": independent,
        "coupled": coupled,
        "delta_test_accuracy_points": (coupled["test_accuracy"] - independent["test_accuracy"]) * 100.0,
        "delta_train_time_sec": coupled["train_time_sec"] - independent["train_time_sec"],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
