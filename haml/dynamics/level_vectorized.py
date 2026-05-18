"""Vectorized level dynamics with backward-compatible attractor views."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.cluster import KMeans


@dataclass
class _AttractorView:
    """Compatibility view exposing a single attractor-like interface."""

    level: "LevelVectorized"
    index: int

    @property
    def position(self):
        return self.level.positions[self.index]

    @property
    def log_sigma(self):
        return self.level.log_sigma[self.index]

    @property
    def log_rho(self):
        return self.level.log_rho[self.index]

    @property
    def log_weight(self):
        return self.level.log_weight[self.index]

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def rho(self):
        return torch.exp(self.log_rho)

    @property
    def weight(self):
        return torch.exp(self.log_weight)

    def enforce_constraints(self):
        with torch.no_grad():
            sigma = self.sigma
            rho = self.rho
            if rho < sigma:
                self.log_rho.copy_(self.log_sigma + 0.1)


class LevelVectorized(nn.Module):
    """Vectorized hierarchical level using flat parameter tensors."""

    def __init__(
        self,
        level_idx,
        dim,
        n_classes,
        n_attractors_per_class,
        lambda_repulsion=0.5,
        rho_sigma_ratio=2.0,
        repulsion_mode="global",
    ):
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim
        self.n_classes = n_classes
        self.n_attractors_per_class = n_attractors_per_class
        self.lambda_repulsion = lambda_repulsion
        self.rho_sigma_ratio = rho_sigma_ratio
        if repulsion_mode not in ("global", "inter_class_only"):
            raise ValueError("repulsion_mode must be 'global' or 'inter_class_only'.")
        self.repulsion_mode = repulsion_mode

        self.n_total = self.n_classes * self.n_attractors_per_class
        self.positions = nn.Parameter(torch.empty(self.n_total, self.dim))
        self.log_sigma = nn.Parameter(torch.empty(self.n_total))
        self.log_rho = nn.Parameter(torch.empty(self.n_total))
        self.log_weight = nn.Parameter(torch.empty(self.n_total))

        class_indices = []
        class_to_indices = []
        for c in range(self.n_classes):
            start = c * self.n_attractors_per_class
            end = start + self.n_attractors_per_class
            class_to_indices.append(torch.arange(start, end, dtype=torch.long))
            class_indices.extend([c] * self.n_attractors_per_class)
        self.register_buffer("class_indices", torch.tensor(class_indices, dtype=torch.long))
        self._class_to_indices = class_to_indices
        self._attractor_views = None

        # Safe defaults before initialize_attractors().
        nn.init.zeros_(self.positions)
        nn.init.zeros_(self.log_sigma)
        nn.init.zeros_(self.log_rho)
        nn.init.zeros_(self.log_weight)

    @property
    def attractors(self):
        # Rebuild lazily to keep references aligned if module is moved/copied.
        if self._attractor_views is None:
            views = {}
            for c in range(self.n_classes):
                c_views = []
                for idx in self._class_to_indices[c].tolist():
                    c_views.append(_AttractorView(level=self, index=idx))
                views[str(c)] = c_views
            self._attractor_views = views
        return self._attractor_views

    def _flat_force(self, x, sigma_sq, sign=1.0):
        delta = self.positions.unsqueeze(0) - x.unsqueeze(1)  # (B, A, D)
        dist_sq = torch.sum(delta ** 2, dim=-1)  # (B, A)
        kernel = torch.exp(-dist_sq / (2.0 * sigma_sq.unsqueeze(0)))  # (B, A)
        coeff = sign * torch.exp(self.log_weight) / sigma_sq  # (A,)
        force = coeff.unsqueeze(0).unsqueeze(-1) * delta * kernel.unsqueeze(-1)
        return force

    def initialize_attractors(self, X, y, sigma_init=None):
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
        else:
            X_np = X
            y_np = y

        pos = torch.empty(self.n_total, self.dim, dtype=torch.float32)
        log_sigma = torch.empty(self.n_total, dtype=torch.float32)
        log_rho = torch.empty(self.n_total, dtype=torch.float32)
        log_weight = torch.empty(self.n_total, dtype=torch.float32)

        for c in range(self.n_classes):
            mask = y_np == c
            X_c = X_np[mask]
            if len(X_c) == 0:
                raise ValueError(f"Cannot initialize level {self.level_idx}: class {c} has no samples.")

            if len(X_c) < self.n_attractors_per_class:
                n_needed = self.n_attractors_per_class
                indices = torch.randint(0, len(X_c), (n_needed,))
                positions_c = torch.from_numpy(X_c[indices]).float()
                positions_c += torch.randn_like(positions_c) * 0.01
            else:
                kmeans = KMeans(n_clusters=self.n_attractors_per_class, n_init=10, random_state=42)
                kmeans.fit(X_c)
                positions_c = torch.from_numpy(kmeans.cluster_centers_).float()

            if sigma_init is None:
                if len(X_c) > 1:
                    data_std = torch.std(torch.from_numpy(X_c).float()).item()
                    sigma = max(math.sqrt(self.dim) * data_std, 0.1)
                else:
                    sigma = math.sqrt(self.dim)
            else:
                sigma = sigma_init

            start = c * self.n_attractors_per_class
            end = start + self.n_attractors_per_class
            pos[start:end] = positions_c
            log_sigma[start:end] = math.log(float(sigma))
            log_rho[start:end] = math.log(float(sigma * self.rho_sigma_ratio))
            log_weight[start:end] = 0.0

        with torch.no_grad():
            self.positions.copy_(pos.to(self.positions.device))
            self.log_sigma.copy_(log_sigma.to(self.log_sigma.device))
            self.log_rho.copy_(log_rho.to(self.log_rho.device))
            self.log_weight.copy_(log_weight.to(self.log_weight.device))

    def intra_level_force(self, x):
        sigma_sq = torch.exp(self.log_sigma) ** 2
        rho_sq = torch.exp(self.log_rho) ** 2

        attraction = self._flat_force(x, sigma_sq=sigma_sq, sign=1.0).sum(dim=1)

        repulsion_all = self._flat_force(x, sigma_sq=rho_sq, sign=-1.0)
        if self.repulsion_mode == "global":
            repulsion = repulsion_all.sum(dim=1)
        else:
            one_hot = torch.nn.functional.one_hot(self.class_indices, num_classes=self.n_classes).float()
            class_repulsion = torch.einsum("bad,ac->bcd", repulsion_all, one_hot)
            repulsion = class_repulsion.sum(dim=1)
        return attraction + self.lambda_repulsion * repulsion

    def energy(self, x):
        delta = self.positions.unsqueeze(0) - x.unsqueeze(1)  # (B, A, D)
        dist_sq = torch.sum(delta ** 2, dim=-1)  # (B, A)
        sigma_sq = torch.exp(self.log_sigma) ** 2
        strength = torch.exp(self.log_weight).unsqueeze(0) * torch.exp(-dist_sq / (2.0 * sigma_sq.unsqueeze(0)))
        return -torch.sum(strength, dim=-1)

    def predict_class_scores(self, x):
        delta = self.positions.unsqueeze(0) - x.unsqueeze(1)  # (B, A, D)
        dist_sq = torch.sum(delta ** 2, dim=-1)  # (B, A)
        sigma_sq = torch.exp(self.log_sigma) ** 2
        strength = torch.exp(self.log_weight).unsqueeze(0) * torch.exp(-dist_sq / (2.0 * sigma_sq.unsqueeze(0)))
        one_hot = torch.nn.functional.one_hot(self.class_indices, num_classes=self.n_classes).float()
        return torch.einsum("ba,ac->bc", strength, one_hot)

    def get_all_attractor_positions(self, class_label=None):
        if class_label is None:
            return self.positions
        idx = self._class_to_indices[class_label].to(self.positions.device)
        return self.positions[idx]

    def separation_energy(self):
        if self.n_total < 2:
            return torch.tensor(0.0, device=self.positions.device)

        dist_sq = torch.cdist(self.positions, self.positions, p=2) ** 2
        class_diff = self.class_indices.unsqueeze(0) != self.class_indices.unsqueeze(1)
        sigma_min = torch.minimum(
            torch.tensor(1.0, device=self.positions.device),
            torch.min(torch.exp(self.log_sigma)),
        )
        penalty = torch.exp(-dist_sq / (2.0 * sigma_min ** 2))
        return (penalty * class_diff.float()).sum()

    def enforce_constraints(self):
        with torch.no_grad():
            sigma = torch.exp(self.log_sigma)
            rho = torch.exp(self.log_rho)
            needs_fix = rho < sigma
            self.log_rho[needs_fix] = self.log_sigma[needs_fix] + 0.1

    def freeze_positions(self):
        # Keep compatibility with trainer soft-landing freeze hook.
        self.positions.requires_grad_(False)
