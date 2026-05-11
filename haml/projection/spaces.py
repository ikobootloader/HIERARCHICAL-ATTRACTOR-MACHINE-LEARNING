"""
Module de projection hiérarchique.

Implémente la tour de sous-espaces X = H_0 → H_1 → ... → H_L
avec projections PCA et pseudo-inverses de Moore-Penrose.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


class HierarchicalSpaces(nn.Module):
    """
    Tour de projections hiérarchiques avec PCA.

    Pour chaque niveau l, maintient :
    - π_l : H_{l-1} → H_l (projection descendante)
    - π_l† : H_l → H_{l-1} (pseudo-inverse, reconstruction montante)

    Propriété garantie : ||π_l||_2 ≤ 1 (contraction)
    """

    def __init__(self, input_dim, level_dims, learn_projections=False):
        """
        Args:
            input_dim (int): Dimension d'entrée (d_0)
            level_dims (list): Dimensions [d_1, d_2, ..., d_L]
            learn_projections (bool): Si True, affine les projections par gradient
        """
        super().__init__()

        self.input_dim = input_dim
        self.level_dims = [input_dim] + list(level_dims)
        self.n_levels = len(self.level_dims)
        self.learn_projections = learn_projections

        # Projections π_l et pseudo-inverses π_l†
        self.projections = nn.ParameterList()
        self.pseudo_inverses = nn.ParameterList()

        # Initialisation vide (sera rempli par fit_pca)
        for l in range(1, self.n_levels):
            d_prev, d_curr = self.level_dims[l-1], self.level_dims[l]

            # π_l : R^{d_prev} → R^{d_curr}
            proj = nn.Parameter(
                torch.eye(d_curr, d_prev),
                requires_grad=learn_projections
            )
            self.projections.append(proj)

            # π_l† : R^{d_curr} → R^{d_prev}
            pinv = nn.Parameter(
                torch.eye(d_prev, d_curr),
                requires_grad=False  # Toujours calculé à partir de proj
            )
            self.pseudo_inverses.append(pinv)

    def fit_pca(self, X):
        """
        Initialise les projections par PCA sur les données.

        Args:
            X (np.ndarray ou torch.Tensor): Données (n_samples, input_dim)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X_current = X

        for l in range(1, self.n_levels):
            d_target = self.level_dims[l]

            # PCA au niveau l
            pca = PCA(n_components=d_target, whiten=False)
            pca.fit(X_current)

            # Composantes principales = lignes de π_l
            # π_l = V^T où V sont les eigenvectors
            proj_matrix = torch.from_numpy(
                pca.components_.astype(np.float32)
            )  # (d_target, d_prev)

            # Mise à jour du paramètre
            with torch.no_grad():
                self.projections[l-1].copy_(proj_matrix)

                # Calcul pseudo-inverse : π_l† = V
                # Pour PCA orthonormale : π_l† = π_l^T
                pinv_matrix = proj_matrix.T  # (d_prev, d_target)
                self.pseudo_inverses[l-1].copy_(pinv_matrix)

            # Projeter les données pour le niveau suivant
            X_current = pca.transform(X_current)

        self._verify_contraction_property()

    def _verify_contraction_property(self):
        """Vérifie que ||π_l||_2 ≤ 1 (condition de stabilité)."""
        for l, proj in enumerate(self.projections):
            norm = torch.linalg.matrix_norm(proj, ord=2).item()
            if norm > 1.05:  # Tolérance numérique
                print(f"Warning: Projection {l+1} has norm {norm:.3f} > 1")

    def project_down(self, x, from_level, to_level):
        """
        Projette x du niveau from_level vers to_level (descendant).

        Args:
            x (torch.Tensor): Vecteur d'état (..., d_from)
            from_level (int): Niveau de départ
            to_level (int): Niveau cible (> from_level)

        Returns:
            torch.Tensor: Projection (..., d_to)
        """
        assert to_level > from_level, "to_level doit être > from_level"

        x_proj = x
        for l in range(from_level, to_level):
            x_proj = x_proj @ self.projections[l].T  # (*, d) @ (d, d') = (*, d')

        return x_proj

    def project_up(self, x, from_level, to_level):
        """
        Reconstruit x du niveau from_level vers to_level (montant).

        Args:
            x (torch.Tensor): Vecteur d'état (..., d_from)
            from_level (int): Niveau de départ
            to_level (int): Niveau cible (< from_level)

        Returns:
            torch.Tensor: Reconstruction (..., d_to)
        """
        assert to_level < from_level, "to_level doit être < from_level"

        x_recon = x
        for l in range(from_level - 1, to_level - 1, -1):
            x_recon = x_recon @ self.pseudo_inverses[l].T  # (*, d) @ (d, d') = (*, d')

        return x_recon

    def forward(self, x):
        """
        Projette x sur tous les niveaux hiérarchiques.

        Args:
            x (torch.Tensor): Données d'entrée (batch, input_dim)

        Returns:
            list[torch.Tensor]: États à chaque niveau [x^(0), x^(1), ..., x^(L)]
        """
        states = [x]  # Niveau 0

        for l in range(1, self.n_levels):
            x_proj = self.project_down(states[-1], l-1, l)
            states.append(x_proj)

        return states

    def get_projection_matrix(self, level):
        """Retourne π_l (niveau level-1 → level)."""
        if level == 0:
            return torch.eye(self.input_dim)
        return self.projections[level - 1]

    def get_pseudoinverse_matrix(self, level):
        """Retourne π_l† (niveau level → level-1)."""
        if level == 0:
            return torch.eye(self.input_dim)
        return self.pseudo_inverses[level - 1]

    def update_pseudoinverses(self):
        """
        Recalcule les pseudo-inverses après mise à jour des projections.
        À appeler si learn_projections=True pendant l'entraînement.
        """
        if not self.learn_projections:
            return

        with torch.no_grad():
            for l in range(len(self.projections)):
                proj = self.projections[l]
                # Pseudo-inverse de Moore-Penrose via SVD
                pinv = torch.linalg.pinv(proj)
                self.pseudo_inverses[l].copy_(pinv)

    def get_contraction_norms(self):
        """Retourne les normes ||π_l||_2 pour vérification."""
        norms = []
        for proj in self.projections:
            norm = torch.linalg.matrix_norm(proj, ord=2).item()
            norms.append(norm)
        return norms
