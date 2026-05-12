"""
Réimplémentation SA-nODE (paper-faithful, version pratique benchmark).

Principe:
- Espace d'état plat (dimension = input)
- Potentiel local double puits analytique
- Attracteurs binaires plantes +/- a
- Couplage lineaire global via matrice A entraînée
"""

import random
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SANodeReimpl(BaseEstimator, ClassifierMixin):
    """
    Réimplémentation SA-nODE pour comparaison reproductible.
    """

    def __init__(
        self,
        a=1.0,
        beta=0.8,
        gamma=0.0,
        dt=0.05,
        n_steps=120,
        lr=0.02,
        n_epochs=120,
        lambda_kernel=2.0,
        lambda_l2=1e-3,
        seed=42,
        device="cpu",
    ):
        self.a = float(a)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.lambda_kernel = float(lambda_kernel)
        self.lambda_l2 = float(lambda_l2)
        self.seed = int(seed)
        self.device = device

        self.classes_ = None
        self.targets_ = None
        self.A_ = None
        self.is_fitted_ = False

    def _build_binary_targets(self, X, y):
        classes = np.unique(y)
        targets = []
        for c in classes:
            m = X[y == c].mean(axis=0)
            sign = np.sign(m)
            sign[sign == 0.0] = 1.0
            targets.append(self.a * sign.astype(np.float32))
        return classes, np.stack(targets, axis=0)

    def _simulate(self, x0, A):
        x = x0
        for _ in range(self.n_steps):
            # dV/dx for V = 1/4 * (x^2 - a^2)^2
            dV = x * (x * x - self.a * self.a)
            dx = -dV + self.beta * (x @ A.T) - self.gamma * x
            x = x + self.dt * dx
        return x

    def fit(self, X, y):
        _set_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes, targets = self._build_binary_targets(X, y)

        self.classes_ = classes
        self.targets_ = targets

        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        T_t = torch.from_numpy(targets).to(self.device)

        d = X.shape[1]
        A = torch.zeros((d, d), dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([A], lr=self.lr)

        # matrix target kernel: A @ phi_k = 0
        phi = T_t.T  # d x K

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            xT = self._simulate(X_t, A)
            target_batch = T_t[y_t]

            loss_fit = torch.mean((xT - target_batch) ** 2)
            kernel_res = A @ phi
            loss_kernel = torch.mean(kernel_res ** 2)
            loss_l2 = torch.mean(A ** 2)

            loss = loss_fit + self.lambda_kernel * loss_kernel + self.lambda_l2 * loss_l2
            loss.backward()
            optimizer.step()

            # symmetrization improves stability of linear term
            with torch.no_grad():
                A.copy_(0.5 * (A + A.T))

        self.A_ = A.detach().cpu().numpy()
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        X = np.asarray(X, dtype=np.float32)
        x = torch.from_numpy(X).to(self.device)
        A = torch.from_numpy(self.A_.astype(np.float32)).to(self.device)
        T = torch.from_numpy(self.targets_.astype(np.float32)).to(self.device)

        with torch.no_grad():
            xT = self._simulate(x, A)
            # class score: negative distance to planted attractor
            # scores shape: n_samples x n_classes
            diff = xT.unsqueeze(1) - T.unsqueeze(0)
            scores = -torch.mean(diff * diff, dim=2)
        return scores.cpu().numpy()

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
