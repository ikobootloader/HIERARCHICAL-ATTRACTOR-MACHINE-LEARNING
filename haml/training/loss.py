"""
Module HAMLLoss.

Implémente la loss composite :
    L = L_CE + μ₁ L_sep + μ₂ L_dyn

- L_CE : Cross-entropy de classification
- L_sep : Séparation inter-classes (maintient condition C3)
- L_dyn : Régularisation dynamique (convergence rapide)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HAMLLoss(nn.Module):
    """
    Loss composite pour l'entraînement HAML.

    L = L_CE + μ₁ L_sep + μ₂ L_dyn

    où :
    - L_CE : -log P(y | {x^(l)(T)})
    - L_sep : Σ_l Σ_{c≠c'} exp(-||μ_{c,m} - μ_{c',m'}||² / (2σ²))
    - L_dyn : (1/T) ∫ Σ_l ||dx^(l)/dt||² dt
    """

    def __init__(self, mu_sep=0.1, mu_dyn=0.01, level_weights=None):
        """
        Args:
            mu_sep (float): Poids μ₁ pour L_sep
            mu_dyn (float): Poids μ₂ pour L_dyn
            level_weights (list, optional): Poids β_l par niveau pour agrégation
                                           (défaut : 2^l, plus de poids aux niveaux abstraits)
        """
        super().__init__()

        self.mu_sep = mu_sep
        self.mu_dyn = mu_dyn
        self.level_weights = level_weights

    def classification_loss(self, final_states, levels, targets):
        """
        Calcule L_CE = -log P(y | {x^(l)(T)}).

        P(c | states) ∝ Σ_l β_l * Σ_m exp(-||x^(l)(T) - μ_{c,m}||² / (2σ²))

        Args:
            final_states (list[torch.Tensor]): États finaux [x^(0)(T), ..., x^(L)(T)]
            levels (list[Level]): Niveaux hiérarchiques
            targets (torch.Tensor): Labels vrais (batch,)

        Returns:
            torch.Tensor: L_CE (scalaire)
        """
        batch_size = final_states[0].shape[0]
        n_classes = levels[0].n_classes
        n_levels = len(levels)

        # Calcul des scores par niveau
        logits = torch.zeros(batch_size, n_classes, device=final_states[0].device)

        for l, (state, level) in enumerate(zip(final_states, levels)):
            # Score d'attraction par classe
            scores = level.predict_class_scores(state)  # (batch, n_classes)

            # Poids du niveau
            if self.level_weights is not None:
                beta = self.level_weights[l]
            else:
                beta = 2 ** l  # Poids exponentiel (niveaux abstraits dominent)

            logits += beta * scores

        # Cross-entropy
        loss = F.cross_entropy(logits, targets)

        return loss

    def separation_loss(self, levels):
        """
        Calcule L_sep pour maintenir la séparation inter-classes.

        L_sep = Σ_l Σ_{c≠c'} Σ_{m,m'} exp(-||μ_{c,m}^(l) - μ_{c',m'}^(l)||² / (2σ_min²))

        Args:
            levels (list[Level]): Niveaux hiérarchiques

        Returns:
            torch.Tensor: L_sep (scalaire)
        """
        loss = torch.tensor(0.0, device=next(levels[0].parameters()).device)

        for level in levels:
            loss += level.separation_energy()

        return loss

    def dynamic_loss(self, trajectory):
        """
        Calcule L_dyn = (1/T) ∫ Σ_l ||dx^(l)/dt||² dt.

        Encourage la convergence rapide et fluide.

        Args:
            trajectory (list): Liste des états à chaque pas de temps
                              [(states_0, ..., states_T)]

        Returns:
            torch.Tensor: L_dyn (scalaire)
        """
        if len(trajectory) < 2:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=trajectory[0][0].device)
        n_steps = len(trajectory) - 1

        for i in range(n_steps):
            states_old = trajectory[i]
            states_new = trajectory[i + 1]

            # Somme sur tous les niveaux
            for x_old, x_new in zip(states_old, states_new):
                dx = x_new - x_old
                loss += torch.sum(dx ** 2)

        # Normalisation par le temps total
        loss = loss / n_steps

        return loss

    def forward(self, final_states, levels, targets, trajectory=None):
        """
        Calcule la loss totale.

        Args:
            final_states (list[torch.Tensor]): États finaux après intégration
            levels (list[Level]): Niveaux hiérarchiques
            targets (torch.Tensor): Labels vrais
            trajectory (list, optional): Trajectoire complète (pour L_dyn)

        Returns:
            dict: {'total': L, 'ce': L_CE, 'sep': L_sep, 'dyn': L_dyn}
        """
        # Loss de classification
        l_ce = self.classification_loss(final_states, levels, targets)

        # Loss de séparation
        l_sep = self.separation_loss(levels)

        # Loss dynamique
        if trajectory is not None and self.mu_dyn > 0:
            l_dyn = self.dynamic_loss(trajectory)
        else:
            l_dyn = torch.tensor(0.0, device=final_states[0].device)

        # Loss totale
        loss_total = l_ce + self.mu_sep * l_sep + self.mu_dyn * l_dyn

        return {
            'total': loss_total,
            'ce': l_ce.item(),
            'sep': l_sep.item(),
            'dyn': l_dyn.item()
        }

    def set_weights(self, mu_sep=None, mu_dyn=None):
        """
        Modifie les poids de la loss composite.

        Utile pour couplage progressif :
        - Phase 1 : μ_sep élevé, μ_dyn = 0 (stabilise attracteurs)
        - Phase 2 : μ_sep maintenu, μ_dyn augmente (ajoute couplage)
        """
        if mu_sep is not None:
            self.mu_sep = mu_sep
        if mu_dyn is not None:
            self.mu_dyn = mu_dyn


class ELBOLoss(nn.Module):
    """
    Loss ELBO alternative pour extension VAE génératif.

    L_ELBO = -E_q[log p(x^(0) | x^(1:L))] + Σ_l KL[q^(l) || p^(l)]

    Non utilisée dans la version actuelle, mais préparée pour extension future.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError("ELBO loss reserved for future VAE extension")

    def forward(self, states, levels):
        """TODO: Implémenter pour modèle génératif."""
        pass
