"""
Benchmark bruité pour HAML.

Compare sur make_moons bruité:
- HAML indépendant (alpha_bu=0, alpha_td=0)
- HAML couplé (alpha_bu=0.5, alpha_td=1.5)
- KNN
- SVM RBF
"""

import json
import random
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from haml import HAML


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_haml(X_train, y_train, X_test, y_test, alpha_bu, alpha_td, seed, n_epochs=20):
    set_seed(seed)
    model = HAML(
        n_levels=3,
        n_attractors_per_class=5,
        alpha_bu=alpha_bu,
        alpha_td=alpha_td,
        max_steps=35,
        lr=0.01,
        n_epochs=n_epochs,
        batch_size=128,
        train_on_fit=False,
        device="cpu",
    )
    model.fit(X_train, y_train)
    model.train_model(X_train, y_train)
    return float(model.score(X_test, y_test))


def run_baselines(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = float(knn.score(X_test, y_test))

    svm = SVC(kernel="rbf", C=10.0, gamma="scale")
    svm.fit(X_train, y_train)
    svm_acc = float(svm.score(X_test, y_test))

    return knn_acc, svm_acc


def main():
    seed = 42
    noise_levels = [0.10, 0.20, 0.30, 0.40]
    n_samples = 3000

    rows = []
    for noise in noise_levels:
        set_seed(seed)
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        haml_ind = run_haml(X_train, y_train, X_test, y_test, 0.0, 0.0, seed=seed, n_epochs=15)
        haml_cpl = run_haml(X_train, y_train, X_test, y_test, 0.5, 1.5, seed=seed, n_epochs=15)
        knn_acc, svm_acc = run_baselines(X_train, y_train, X_test, y_test)

        rows.append(
            {
                "noise": noise,
                "haml_independent": haml_ind,
                "haml_coupled": haml_cpl,
                "knn": knn_acc,
                "svm_rbf": svm_acc,
                "delta_coupled_minus_independent_points": (haml_cpl - haml_ind) * 100.0,
            }
        )

    summary = {
        "dataset": "make_moons",
        "n_samples": n_samples,
        "seed": seed,
        "results": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
