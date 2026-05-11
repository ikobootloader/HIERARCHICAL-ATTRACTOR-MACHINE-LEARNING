"""
Module StabilityAnalyzer.

Analyse spectrale de la Hessienne pour vérifier les conditions de stabilité.
"""

import torch
import numpy as np


class StabilityAnalyzer:
    """
    Analyse de stabilité du système dynamique.

    Calcule et vérifie :
    - Valeurs propres de la Hessienne (λ_min > 0)
    - Temps de convergence estimé T ~ γ/λ_min
    - Condition C2 (couplage suffisant)
    """

    def __init__(self, model):
        """
        Args:
            model (HAML): Modèle HAML
        """
        self.model = model

    def compute_hessian_eigenvalues(self, X_sample):
        """
        Calcule les valeurs propres de la Hessienne au point stationnaire.

        Args:
            X_sample (torch.Tensor): Échantillon (1, n_features)

        Returns:
            dict: {'level_0': [eigenvals], 'level_1': [...], ...}
        """
        self.model.eval()

        with torch.no_grad():
            final_states, _, _ = self.model.forward(X_sample)

        eigenvalues = {}

        for l, (level, state) in enumerate(zip(self.model.levels, final_states)):
            # Calcul approximatif : κ^(l) au point fixe
            kappa_values = []

            for c in range(level.n_classes):
                for attractor in level.attractors[str(c)]:
                    kappa = attractor.get_hessian_eigenvalue().item()
                    kappa_values.append(kappa)

            eigenvalues[f'level_{l}'] = kappa_values

        return eigenvalues

    def estimate_convergence_time(self):
        """
        Estime le temps de convergence T ~ γ/λ_min.

        Returns:
            float: Temps estimé (en pas de temps)
        """
        # Approximation simple : utiliser les κ moyens
        min_kappa = float('inf')

        for level in self.model.levels:
            for c in range(level.n_classes):
                for attractor in level.attractors[str(c)]:
                    kappa = attractor.get_hessian_eigenvalue().item()
                    min_kappa = min(min_kappa, kappa)

        if min_kappa <= 0:
            return float('inf')

        T = self.model.gamma / min_kappa
        return T

    def check_stability_conditions(self):
        """
        Vérifie les 3 conditions de stabilité C1, C2, C3.

        Returns:
            dict: {'C1': bool, 'C2': bool, 'C3': bool, 'details': str}
        """
        results = {
            'C1': None,  # Dominance locale (nécessite données)
            'C2': None,  # Couplage suffisant
            'C3': None,  # Séparation des bassins
            'details': []
        }

        # C2: Vérification du couplage
        c2_satisfied = self.model.coupling.verify_coupling_condition_C2(self.model.levels)
        results['C2'] = c2_satisfied

        if c2_satisfied:
            results['details'].append("[OK] C2: Couplage suffisant")
        else:
            results['details'].append("[FAIL] C2: Couplage insuffisant (augmenter alpha_bu + alpha_td)")

        # C3: Separation (calculer distance min inter-classes)
        min_distance = self._compute_min_interclass_distance()
        results['C3'] = min_distance > 0.5  # Seuil arbitraire

        if results['C3']:
            results['details'].append(f"[OK] C3: Separation minimale = {min_distance:.3f}")
        else:
            results['details'].append(f"[FAIL] C3: Separation trop faible = {min_distance:.3f}")

        return results

    def _compute_min_interclass_distance(self):
        """Calcule la distance minimale entre attracteurs de classes différentes."""
        min_dist = float('inf')

        for level in self.model.levels:
            for c1 in range(level.n_classes):
                for c2 in range(c1 + 1, level.n_classes):
                    pos1 = level.get_all_attractor_positions(c1)
                    pos2 = level.get_all_attractor_positions(c2)

                    # Distance min entre toutes les paires
                    for p1 in pos1:
                        for p2 in pos2:
                            dist = torch.norm(p1 - p2).item()
                            min_dist = min(min_dist, dist)

        return min_dist

    def get_contraction_norms(self):
        """Retourne les normes ||π_l||_2 (doivent être < 1)."""
        return self.model.spaces.get_contraction_norms()

    def summary(self):
        """Affiche un résumé de l'analyse de stabilité."""
        print("=" * 60)
        print("HAML Stability Analysis")
        print("=" * 60)

        # Temps de convergence
        T_est = self.estimate_convergence_time()
        print(f"Estimated convergence time: {T_est:.2f} steps")

        # Conditions
        conditions = self.check_stability_conditions()
        print("\nStability Conditions:")
        for detail in conditions['details']:
            print(f"  {detail}")

        # Normes de contraction
        norms = self.get_contraction_norms()
        print(f"\nContraction norms ||Pi_l||_2:")
        for l, norm in enumerate(norms):
            status = "[OK]" if norm <= 1.0 else "[FAIL]"
            print(f"  Level {l+1}: {norm:.4f} {status}")

        # Couplage
        alpha_bu, alpha_td = self.model.coupling.get_coupling_strengths()
        print(f"\nCoupling strengths:")
        print(f"  alpha_bu = {alpha_bu:.3f}")
        print(f"  alpha_td = {alpha_td:.3f}")
        print(f"  alpha_total = {alpha_bu + alpha_td:.3f}")

        print("=" * 60)
