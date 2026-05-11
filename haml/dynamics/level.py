"""
Module Level.

Implémente un niveau hiérarchique complet avec K×M attracteurs et le calcul de F_intra.
"""

import torch
import torch.nn as nn
from .attractor import Attractor
from sklearn.cluster import KMeans


class Level(nn.Module):
    """
    Niveau hiérarchique avec attracteurs multi-classes.

    Gère :
    - Ensemble d'attracteurs {μ_{c,m}} pour c ∈ [K], m ∈ [M]
    - Calcul de F_intra = gradient de E^(l)
    - Énergie totale E^(l)
    """

    def __init__(self, level_idx, dim, n_classes, n_attractors_per_class, lambda_repulsion=0.5):
        """
        Args:
            level_idx (int): Indice du niveau (0 = input, L = abstract)
            dim (int): Dimension d_l du sous-espace
            n_classes (int): Nombre de classes K
            n_attractors_per_class (int): Nombre d'attracteurs M par classe
            lambda_repulsion (float): Coefficient λ (attraction/répulsion)
        """
        super().__init__()

        self.level_idx = level_idx
        self.dim = dim
        self.n_classes = n_classes
        self.n_attractors_per_class = n_attractors_per_class
        self.lambda_repulsion = lambda_repulsion

        # Attracteurs organisés par classe
        self.attractors = nn.ModuleDict()
        for c in range(n_classes):
            self.attractors[str(c)] = nn.ModuleList()

    def initialize_attractors(self, X, y, sigma_init=None):
        """
        Initialise les attracteurs par k-means sur les données projetées.

        Args:
            X (torch.Tensor): Données projetées au niveau l (n_samples, dim)
            y (torch.Tensor): Labels (n_samples,)
            sigma_init (float, optional): Portée initiale (auto si None)
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
        else:
            X_np = X
            y_np = y

        for c in range(self.n_classes):
            # Données de la classe c
            mask = y_np == c
            X_c = X_np[mask]

            if len(X_c) < self.n_attractors_per_class:
                # Pas assez de données : dupliquer avec bruit
                n_needed = self.n_attractors_per_class
                indices = torch.randint(0, len(X_c), (n_needed,))
                positions = torch.from_numpy(X_c[indices]).float()
                positions += torch.randn_like(positions) * 0.01
            else:
                # K-means pour trouver M attracteurs
                kmeans = KMeans(
                    n_clusters=self.n_attractors_per_class,
                    n_init=10,
                    random_state=42
                )
                kmeans.fit(X_c)
                positions = torch.from_numpy(kmeans.cluster_centers_).float()

            # Estimation de σ : écart-type local rescalé par √d (curse of dimensionality)
            if sigma_init is None:
                if len(X_c) > 1:
                    data_std = torch.std(torch.from_numpy(X_c).float()).item()
                    # Rescaling critique pour haute dimension: σ ~ √d * data_std
                    import math
                    sigma = math.sqrt(self.dim) * data_std
                    sigma = max(sigma, 0.1)  # Borne inf
                else:
                    sigma = math.sqrt(self.dim)
            else:
                sigma = sigma_init

            # Créer les attracteurs
            for m in range(self.n_attractors_per_class):
                attractor = Attractor(
                    position=positions[m],
                    class_label=c,
                    dim=self.dim,
                    sigma_init=sigma,
                    rho_init=sigma * 2.0,  # ρ = 2σ
                    weight_init=1.0
                )
                self.attractors[str(c)].append(attractor)

    def intra_level_force(self, x):
        """
        Calcule F_intra(x) = -∇E^(l)(x).

        F_intra = Σ F_att(propre classe) - λ Σ F_rep(autres classes)

        Args:
            x (torch.Tensor): État au niveau l (..., dim)

        Returns:
            torch.Tensor: Force intra-niveau (..., dim)
        """
        force = torch.zeros_like(x)

        # Pour chaque classe
        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                # Attraction de tous les attracteurs
                force += attractor.attraction_force(x)

                # Répulsion des attracteurs d'autres classes
                # Note : appliquée comme terme négatif dans F_att
                # Ici on les traite séparément via lambda_repulsion

        # Application du terme répulsif (simplifié)
        # Dans la spec complète, il faudrait distinguer classe par classe
        # Ici on applique une répulsion globale pondérée par λ
        force_rep = torch.zeros_like(x)
        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                force_rep += attractor.repulsion_force(x)

        force = force + self.lambda_repulsion * force_rep

        return force

    def energy(self, x):
        """
        Calcule E^(l)(x) = -Σ w * exp(...) + λ * termes répulsifs.

        Args:
            x (torch.Tensor): État (..., dim)

        Returns:
            torch.Tensor: Énergie (...)
        """
        energy = torch.zeros(x.shape[:-1], device=x.device)

        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                energy += attractor.energy(x)

        # Terme répulsif (approximation)
        # Dans la spec: contribution négative aux attracteurs autres classes
        # Ici on l'omet pour simplifier (inclus dans F_intra)

        return energy

    def predict_class_scores(self, x):
        """
        Calcule les scores d'attraction par classe (pour prédiction).

        Score(c) = Σ_m w_{c,m} * exp(-||x - μ_{c,m}||²/(2σ²))

        Args:
            x (torch.Tensor): État (..., dim)

        Returns:
            torch.Tensor: Scores (..., n_classes)
        """
        batch_shape = x.shape[:-1]
        scores = torch.zeros(*batch_shape, self.n_classes, device=x.device)

        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                scores[..., c] += attractor.attraction_strength(x)

        return scores

    def get_all_attractor_positions(self, class_label=None):
        """
        Retourne les positions de tous les attracteurs.

        Args:
            class_label (int, optional): Si spécifié, retourne uniquement classe c

        Returns:
            torch.Tensor: Positions (n_attractors, dim)
        """
        positions = []

        if class_label is not None:
            for attractor in self.attractors[str(class_label)]:
                positions.append(attractor.position)
        else:
            for c in range(self.n_classes):
                for attractor in self.attractors[str(c)]:
                    positions.append(attractor.position)

        return torch.stack(positions)

    def separation_energy(self):
        """
        Calcule la loss de séparation L_sep pour maintenir la condition C3.

        L_sep = Σ_{c≠c'} Σ_{m,m'} exp(-||μ_{c,m} - μ_{c',m'}||² / (2σ_min²))

        Returns:
            torch.Tensor: L_sep (scalaire)
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        # Récupérer toutes les positions et leurs classes
        all_positions = []
        all_classes = []

        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                all_positions.append(attractor.position)
                all_classes.append(c)

        if len(all_positions) < 2:
            return loss

        positions = torch.stack(all_positions)  # (n_total, dim)
        classes = torch.tensor(all_classes, device=positions.device)

        # Calcul des distances par paires
        dist_sq = torch.cdist(positions, positions, p=2) ** 2  # (n, n)

        # Masque inter-classes
        class_diff = classes.unsqueeze(0) != classes.unsqueeze(1)  # (n, n)

        # σ_min = min des sigmas
        sigma_min = torch.tensor(1.0, device=positions.device)
        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                sigma_min = torch.min(sigma_min, attractor.sigma)

        # Pénalité de proximité inter-classes
        penalty = torch.exp(-dist_sq / (2 * sigma_min ** 2))
        loss = (penalty * class_diff.float()).sum()

        return loss

    def enforce_constraints(self):
        """Applique les contraintes sur tous les attracteurs."""
        for c in range(self.n_classes):
            for attractor in self.attractors[str(c)]:
                attractor.enforce_constraints()
