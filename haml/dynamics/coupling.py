"""
Module BidirectionalCoupling.

Implémente les forces de couplage inter-niveaux :
- F_bu : erreur de prédiction bottom-up
- F_td : prédiction top-down
"""

import torch
import torch.nn as nn


class BidirectionalCoupling(nn.Module):
    """
    Calcule les forces de couplage entre niveaux hiérarchiques.

    F_bu^(l) = α_bu * π_l(x^(l-1) - π_l†(x^(l)))
    F_td^(l) = α_td * (π_l†(x^(l+1)) - x^(l))

    Ces termes correspondent exactement au Predictive Coding :
    - F_bu : erreur de prédiction ascendante (ce que le niveau inférieur voit vs ce que je prédis)
    - F_td : prédiction descendante (ce que le niveau supérieur me dit de chercher)
    """

    def __init__(self, hierarchical_spaces, alpha_bu=1.0, alpha_td=1.0, learn_alphas=False):
        """
        Args:
            hierarchical_spaces (HierarchicalSpaces): Tour de projections
            alpha_bu (float): Coefficient bottom-up
            alpha_td (float): Coefficient top-down
            learn_alphas (bool): Si True, α_bu et α_td sont appris
        """
        super().__init__()

        self.spaces = hierarchical_spaces
        self.n_levels = hierarchical_spaces.n_levels

        # Coefficients de couplage
        if learn_alphas:
            self.log_alpha_bu = nn.Parameter(torch.log(torch.tensor(alpha_bu)))
            self.log_alpha_td = nn.Parameter(torch.log(torch.tensor(alpha_td)))
        else:
            self.register_buffer('log_alpha_bu', torch.log(torch.tensor(alpha_bu)))
            self.register_buffer('log_alpha_td', torch.log(torch.tensor(alpha_td)))

    @property
    def alpha_bu(self):
        """Coefficient bottom-up α_bu > 0."""
        return torch.exp(self.log_alpha_bu)

    @property
    def alpha_td(self):
        """Coefficient top-down α_td > 0."""
        return torch.exp(self.log_alpha_td)

    def bottom_up_force(self, x_below, x_current, level):
        """
        Calcule F_bu^(l) = α_bu * π_l(x^(l-1) - π_l†(x^(l))).

        Erreur de prédiction : le niveau l-1 envoie x^(l-1), mais si je reconstruis
        x^(l) en descendant, j'obtiens π_l†(x^(l)). L'écart est l'erreur.

        Args:
            x_below (torch.Tensor): État x^(l-1) (..., d_{l-1})
            x_current (torch.Tensor): État x^(l) (..., d_l)
            level (int): Niveau l

        Returns:
            torch.Tensor: Force bottom-up (..., d_l)
        """
        if level == 0:
            # Pas de niveau en-dessous
            return torch.zeros_like(x_current)

        # Reconstruction du niveau l-1 depuis l
        x_recon = self.spaces.project_up(x_current, level, level - 1)  # (..., d_{l-1})

        # Erreur de prédiction
        prediction_error = x_below - x_recon  # (..., d_{l-1})

        # Projection vers le niveau l
        error_proj = self.spaces.project_down(prediction_error, level - 1, level)  # (..., d_l)

        force = self.alpha_bu * error_proj
        return force

    def top_down_force(self, x_current, x_above, level):
        """
        Calcule F_td^(l) = α_td * (π_l†(x^(l+1)) - x^(l)).

        Prédiction : le niveau l+1 me dit où je devrais être (π_l†(x^(l+1))).
        La force me pousse vers cette prédiction.

        Args:
            x_current (torch.Tensor): État x^(l) (..., d_l)
            x_above (torch.Tensor): État x^(l+1) (..., d_{l+1})
            level (int): Niveau l

        Returns:
            torch.Tensor: Force top-down (..., d_l)
        """
        if level == self.n_levels - 1:
            # Pas de niveau au-dessus
            return torch.zeros_like(x_current)

        # Prédiction depuis le niveau l+1
        x_pred = self.spaces.project_up(x_above, level + 1, level)  # (..., d_l)

        # Force de rappel vers la prédiction
        force = self.alpha_td * (x_pred - x_current)

        return force

    def coupling_energy(self, states):
        """
        Calcule l'énergie de couplage E_coupling.

        E_coupling = (α_bu + α_td)/2 * Σ_l ||x^(l) - π_l†(x^(l+1))||²

        Args:
            states (list[torch.Tensor]): États [x^(0), ..., x^(L)]

        Returns:
            torch.Tensor: Énergie de couplage (scalaire)
        """
        energy = torch.tensor(0.0, device=states[0].device)

        for l in range(self.n_levels - 1):
            x_current = states[l]
            x_above = states[l + 1]

            # Reconstruction de x^(l) depuis x^(l+1)
            x_recon = self.spaces.project_up(x_above, l + 1, l)

            # Écart de reconstruction
            diff = x_current - x_recon
            energy += torch.sum(diff ** 2)

        energy = 0.5 * (self.alpha_bu + self.alpha_td) * energy
        return energy

    def total_force(self, states, level):
        """
        Calcule F_bu + F_td pour un niveau donné.

        Args:
            states (list[torch.Tensor]): États à tous les niveaux
            level (int): Niveau cible

        Returns:
            torch.Tensor: Force de couplage totale
        """
        x_current = states[level]

        # Bottom-up
        if level > 0:
            x_below = states[level - 1]
            f_bu = self.bottom_up_force(x_below, x_current, level)
        else:
            f_bu = torch.zeros_like(x_current)

        # Top-down
        if level < self.n_levels - 1:
            x_above = states[level + 1]
            f_td = self.top_down_force(x_current, x_above, level)
        else:
            f_td = torch.zeros_like(x_current)

        return f_bu + f_td

    def get_coupling_strengths(self):
        """Retourne (α_bu, α_td) pour monitoring."""
        return self.alpha_bu.item(), self.alpha_td.item()

    def verify_coupling_condition_C2(self, levels):
        """
        Vérifie la condition C2 : α_bu + α_td > α_min.

        α_min est la borne inférieure pour garantir cohérence inter-niveaux.
        Calcul approximatif : α_min ≈ max_l ||π_l||_2

        Args:
            levels (list[Level]): Niveaux hiérarchiques

        Returns:
            bool: True si C2 est vérifiée
        """
        alpha_total = self.alpha_bu + self.alpha_td

        # Borne approximative
        norms = self.spaces.get_contraction_norms()
        alpha_min = max(norms) if norms else 0.5

        is_satisfied = alpha_total > alpha_min

        if not is_satisfied:
            print(
                f"Warning: C2 not satisfied. alpha_bu + alpha_td = "
                f"{alpha_total:.3f} < alpha_min = {alpha_min:.3f}"
            )

        return is_satisfied
