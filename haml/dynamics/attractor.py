"""
Module Attractor.

Implémente un attracteur individuel μ_{c,m} avec ses 4 paramètres appris :
- position μ
- portée d'attraction σ
- portée de répulsion ρ
- poids w
"""

import torch
import torch.nn as nn


class Attractor(nn.Module):
    """
    Attracteur gaussien individuel.

    Force d'attraction :
        F_att(x) = (w/σ²)(μ - x) * exp(-||x - μ||²/(2σ²))

    Force de répulsion (vers attracteurs d'autres classes) :
        F_rep(x) = -(w/ρ²)(μ - x) * exp(-||x - μ||²/(2ρ²))
    """

    def __init__(self, position, class_label, dim, sigma_init=1.0, rho_init=2.0, weight_init=1.0):
        """
        Args:
            position (torch.Tensor): Position initiale μ (dim,)
            class_label (int): Classe c de l'attracteur
            dim (int): Dimension de l'espace
            sigma_init (float): Portée d'attraction initiale
            rho_init (float): Portée de répulsion initiale (> sigma)
            weight_init (float): Poids initial
        """
        super().__init__()

        self.class_label = class_label
        self.dim = dim

        # Paramètres appris
        self.position = nn.Parameter(position.clone())
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))
        self.log_rho = nn.Parameter(torch.log(torch.tensor(rho_init)))
        self.log_weight = nn.Parameter(torch.log(torch.tensor(weight_init)))

    @property
    def sigma(self):
        """Portée d'attraction σ > 0."""
        return torch.exp(self.log_sigma)

    @property
    def rho(self):
        """Portée de répulsion ρ > 0."""
        return torch.exp(self.log_rho)

    @property
    def weight(self):
        """Poids w > 0."""
        return torch.exp(self.log_weight)

    def attraction_force(self, x):
        """
        Calcule la force d'attraction F_att(x) = (w/σ²)(μ - x) * K_σ(x).

        Args:
            x (torch.Tensor): État (..., dim)

        Returns:
            torch.Tensor: Force (..., dim)
        """
        delta = self.position - x  # (..., dim)
        dist_sq = torch.sum(delta ** 2, dim=-1, keepdim=True)  # (..., 1)

        sigma_sq = self.sigma ** 2
        kernel = torch.exp(-dist_sq / (2 * sigma_sq))  # (..., 1)

        force = (self.weight / sigma_sq) * delta * kernel  # (..., dim)
        return force

    def repulsion_force(self, x):
        """
        Calcule la force de répulsion F_rep(x) = -(w/ρ²)(μ - x) * K_ρ(x).

        Args:
            x (torch.Tensor): État (..., dim)

        Returns:
            torch.Tensor: Force (..., dim)
        """
        delta = self.position - x  # (..., dim)
        dist_sq = torch.sum(delta ** 2, dim=-1, keepdim=True)  # (..., 1)

        rho_sq = self.rho ** 2
        kernel = torch.exp(-dist_sq / (2 * rho_sq))  # (..., 1)

        force = -(self.weight / rho_sq) * delta * kernel  # (..., dim)
        return force

    def energy(self, x):
        """
        Calcule la contribution à l'énergie : -w * exp(-||x - μ||²/(2σ²)).

        Args:
            x (torch.Tensor): État (..., dim)

        Returns:
            torch.Tensor: Énergie (...)
        """
        delta = self.position - x
        dist_sq = torch.sum(delta ** 2, dim=-1)  # (...)

        sigma_sq = self.sigma ** 2
        energy = -self.weight * torch.exp(-dist_sq / (2 * sigma_sq))

        return energy

    def attraction_strength(self, x):
        """
        Calcule la force d'attraction normalisée (pour prédiction).

        Args:
            x (torch.Tensor): État (..., dim)

        Returns:
            torch.Tensor: Force scalaire (...) ≥ 0
        """
        delta = self.position - x
        dist_sq = torch.sum(delta ** 2, dim=-1)  # (...)

        sigma_sq = self.sigma ** 2
        strength = self.weight * torch.exp(-dist_sq / (2 * sigma_sq))

        return strength

    def distance_to(self, x):
        """Calcule ||x - μ||."""
        delta = self.position - x
        return torch.norm(delta, dim=-1)

    def enforce_constraints(self):
        """
        Applique les contraintes :
        - ρ > σ (répulsion plus large que attraction)
        - Optionnel : borne supérieure sur σ, ρ
        """
        with torch.no_grad():
            # Contrainte ρ > σ
            if self.rho < self.sigma:
                self.log_rho.copy_(self.log_sigma + 0.1)

    def get_hessian_eigenvalue(self):
        """
        Calcule la valeur propre κ de la Hessienne au point fixe x = μ.

        κ = w/σ² - λ * Σ_{c' ≠ c} (w'/ρ'²) * exp(-||μ - μ'||²/(2ρ'²))

        Note : nécessite accès aux autres attracteurs, donc calculé au niveau Level.

        Returns:
            torch.Tensor: κ (scalaire)
        """
        # Contribution locale uniquement (terme w/σ²)
        kappa_local = self.weight / (self.sigma ** 2)
        return kappa_local
