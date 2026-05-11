"""
Module ConstrainedOptimizer.

Wrapper autour d'Adam qui applique les contraintes de stabilité après chaque step.
"""

import torch
from torch.optim import Adam


class ConstrainedOptimizer:
    """
    Optimiseur avec projection gradient pour maintenir les contraintes.

    Contraintes appliquées :
    - ρ > σ (répulsion plus large que attraction)
    - Séparation minimale inter-classes (optionnel)
    """

    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        """
        Args:
            parameters: Paramètres du modèle
            lr (float): Learning rate
            weight_decay (float): Régularisation L2
        """
        self.optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
        self.lr = lr

    def zero_grad(self):
        """Reset gradients."""
        self.optimizer.zero_grad()

    def step(self, model):
        """
        Effectue un pas d'optimisation et applique les contraintes.

        Args:
            model (HAML): Modèle HAML
        """
        # Step Adam standard
        self.optimizer.step()

        # Application des contraintes
        self._enforce_constraints(model)

        # Mise à jour des pseudo-inverses si projections apprises
        if model.spaces.learn_projections:
            model.spaces.update_pseudoinverses()

    def _enforce_constraints(self, model):
        """
        Applique les contraintes sur tous les attracteurs.

        Args:
            model (HAML): Modèle HAML
        """
        with torch.no_grad():
            for level in model.levels:
                level.enforce_constraints()

    def state_dict(self):
        """State dict pour sauvegarde."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Charge state dict."""
        self.optimizer.load_state_dict(state_dict)
