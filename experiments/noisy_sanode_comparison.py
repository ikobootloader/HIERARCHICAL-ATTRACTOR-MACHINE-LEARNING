"""
Benchmark bruité avec comparaison HAML vs SA-nODE-like.

Protocole:
- Dataset: make_moons
- Seed fixe et split fixe
- Niveaux de bruit identiques

Modèles comparés:
- HAML indépendant (alpha_bu=0, alpha_td=0)
- HAML couplé (alpha_bu=0.5, alpha_td=1.5)
- SA-nODE-like (espace unique + dynamique double-puits)
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


class SANodeLikeClassifier:
    """
    Baseline SA-nODE-like:
    - espace unique (pas de hiérarchie)
    - attracteurs de signe binaire appris depuis les classes
    - dynamique dissipative sur potentiel double-puits
    """

    def __init__(self, a=1.0, beta=2.0, dt=0.05, n_steps=120):
        self.a = float(a)
        self.beta = float(beta)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.class_signatures_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        signatures = {}
        for c in self.classes_:
            mean_c = X[y == c].mean(axis=0)
            sign = np.sign(mean_c)
            sign[sign == 0.0] = 1.0
            signatures[int(c)] = sign.astype(np.float64)
        self.class_signatures_ = signatures
        return self

    def _run_dynamics(self, X):
        x = np.asarray(X, dtype=np.float64).copy()
        for _ in range(self.n_steps):
            # Gradient du potentiel quartique: d/dx (x^2 - a^2)^2 = 4x(x^2 - a^2)
            grad = 4.0 * x * (x * x - self.a * self.a)
            x -= self.dt * grad
        return x

    def decision_function(self, X):
        x_final = self._run_dynamics(X)
        scores = np.zeros((x_final.shape[0], len(self.classes_)), dtype=np.float64)
        for idx, c in enumerate(self.classes_):
            s = self.class_signatures_[int(c)]
            # Score = alignement après relaxation + régularisation d'amplitude
            align = (x_final * s).sum(axis=1)
            amp = np.abs(np.abs(x_final) - self.a).mean(axis=1)
            scores[:, idx] = align - self.beta * amp
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return float((y_pred == y).mean())


def run_haml(X_train, y_train, X_test, y_test, alpha_bu, alpha_td, seed, n_epochs=15):
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
    sa_node_like = SANodeLikeClassifier(a=1.0, beta=2.0, dt=0.05, n_steps=120)
    sa_node_like.fit(X_train, y_train)
    sa_node_like_acc = float(sa_node_like.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = float(knn.score(X_test, y_test))

    svm = SVC(kernel="rbf", C=10.0, gamma="scale")
    svm.fit(X_train, y_train)
    svm_acc = float(svm.score(X_test, y_test))

    return sa_node_like_acc, knn_acc, svm_acc


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
        sa_node_like_acc, knn_acc, svm_acc = run_baselines(X_train, y_train, X_test, y_test)

        rows.append(
            {
                "noise": noise,
                "haml_independent": haml_ind,
                "haml_coupled": haml_cpl,
                "sa_node_like": sa_node_like_acc,
                "knn": knn_acc,
                "svm_rbf": svm_acc,
                "delta_coupled_minus_independent_points": (haml_cpl - haml_ind) * 100.0,
                "delta_coupled_minus_sa_node_like_points": (haml_cpl - sa_node_like_acc) * 100.0,
            }
        )

    summary = {
        "dataset": "make_moons",
        "n_samples": n_samples,
        "seed": seed,
        "notes": "SA-nODE-like baseline: single-space double-well dynamics with binary class signatures.",
        "results": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
