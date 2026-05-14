"""
Module HAML principal.

Orchestrateur du système hiérarchique d'attracteurs.
Interface compatible scikit-learn (fit/predict).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from ..projection import HierarchicalSpaces
from ..dynamics import Level, BidirectionalCoupling, ODEIntegrator
from ..training import (
    HAMLLoss,
    ConstrainedOptimizer,
    PhaseConfig,
    compute_level_accuracy,
    compute_attraction_force_stats
)
from ..training.trainer import HAMLTrainer


class HAML(nn.Module, BaseEstimator, ClassifierMixin):
    """
    Hierarchical Attractor Machine Learning.

    Système de classification basé sur une dynamique hiérarchique bidirectionnelle
    où la prédiction émerge de la convergence vers un bassin d'attraction stable.

    Architecture :
        X = H_0 → H_1 → ... → H_L (projections PCA)
        Attracteurs {μ_{c,m}^(l)} à chaque niveau
        Couplage bidirectionnel (bottom-up + top-down)
        Intégration ODE jusqu'à régime stationnaire
    """

    def __init__(
        self,
        n_levels=3,
        level_dims=None,
        n_attractors_per_class=5,
        alpha_bu=1.0,
        alpha_td=1.0,
        gamma=1.0,
        lambda_repulsion=0.5,
        rho_sigma_ratio=2.0,
        mu_sep=0.1,
        mu_dyn=0.01,
        integrator_method='rk4',
        dt=0.1,
        max_steps=100,
        tol=1e-4,
        learn_projections=False,
        learn_alphas=False,
        lr=0.01,
        n_epochs=50,
        batch_size=32,
        train_on_fit=False,
        level_score_weighting='exponential',
        device='cpu'
    ):
        """
        Args:
            n_levels (int): Nombre de niveaux L
            level_dims (list, optional): Dimensions [d_1, ..., d_L] (auto si None)
            n_attractors_per_class (int): Nombre M d'attracteurs par classe
            alpha_bu (float): Coefficient bottom-up
            alpha_td (float): Coefficient top-down
            gamma (float): Amortissement
            lambda_repulsion (float): Ratio attraction/répulsion λ
            rho_sigma_ratio (float): Ratio d'initialisation rho/sigma (>1.0)
            mu_sep (float): Poids L_sep
            mu_dyn (float): Poids L_dyn
            integrator_method (str): 'euler', 'rk4', ou 'adjoint'
            dt (float): Pas de temps
            max_steps (int): Max itérations ODE
            tol (float): Tolérance convergence
            learn_projections (bool): Affiner projections PCA
            learn_alphas (bool): Apprendre α_bu, α_td
            level_score_weighting (str): 'uniform' ou 'exponential'
            device (str): 'cpu' ou 'cuda'
        """
        super().__init__()

        self.n_levels = n_levels
        self.level_dims = level_dims
        self.n_attractors_per_class = n_attractors_per_class
        self.device = device

        # Hyperparamètres stockés (pour scikit-learn)
        self.alpha_bu = alpha_bu
        self.alpha_td = alpha_td
        self.gamma = gamma
        self.lambda_repulsion = lambda_repulsion
        self.rho_sigma_ratio = rho_sigma_ratio
        self.mu_sep = mu_sep
        self.mu_dyn = mu_dyn
        self.integrator_method = integrator_method
        self.dt = dt
        self.max_steps = max_steps
        self.tol = tol
        self.learn_projections = learn_projections
        self.learn_alphas = learn_alphas
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_on_fit = train_on_fit
        self.level_score_weighting = level_score_weighting

        # Modules (initialisés dans fit)
        self.scaler = StandardScaler()
        self.spaces = None
        self.levels = None
        self.coupling = None
        self.integrator = None
        self.loss_fn = None

        self.n_classes_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def _initialize_architecture(self, X, y):
        """
        Initialise la hiérarchie après avoir vu les données.

        Args:
            X (np.ndarray): Données (n_samples, n_features)
            y (np.ndarray): Labels (n_samples,)
        """
        n_samples, input_dim = X.shape
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        # Dimensions par niveau (auto si non spécifié)
        if self.level_dims is None:
            # Décroissance géométrique
            dims = []
            d = input_dim
            for l in range(1, self.n_levels):
                d = max(2, d // 2)  # Min 2D
                dims.append(d)
            self.level_dims = dims
        else:
            assert len(self.level_dims) == self.n_levels - 1

        print(f"Architecture: {input_dim} -> {' -> '.join(map(str, self.level_dims))}")

        # 1. Tour de sous-espaces
        self.spaces = HierarchicalSpaces(
            input_dim=input_dim,
            level_dims=self.level_dims,
            learn_projections=self.learn_projections
        ).to(self.device)

        # Initialisation PCA
        X_torch = torch.from_numpy(X).float().to(self.device)
        self.spaces.fit_pca(X_torch)

        # 2. Niveaux hiérarchiques
        all_dims = [input_dim] + self.level_dims
        self.levels = nn.ModuleList()

        for l, dim in enumerate(all_dims):
            level = Level(
                level_idx=l,
                dim=dim,
                n_classes=self.n_classes_,
                n_attractors_per_class=self.n_attractors_per_class,
                lambda_repulsion=self.lambda_repulsion,
                rho_sigma_ratio=self.rho_sigma_ratio
            ).to(self.device)
            self.levels.append(level)

        # Initialisation des attracteurs
        states = self.spaces(X_torch)
        y_torch = torch.from_numpy(y).long().to(self.device)

        for l, (level, state) in enumerate(zip(self.levels, states)):
            level.initialize_attractors(state, y_torch)

        # 3. Couplage bidirectionnel
        self.coupling = BidirectionalCoupling(
            hierarchical_spaces=self.spaces,
            alpha_bu=self.alpha_bu,
            alpha_td=self.alpha_td,
            learn_alphas=self.learn_alphas
        ).to(self.device)

        # 4. Intégrateur ODE
        self.integrator = ODEIntegrator(
            levels=list(self.levels),
            coupling=self.coupling,
            gamma=self.gamma,
            method=self.integrator_method,
            dt=self.dt,
            max_steps=self.max_steps,
            tol=self.tol
        ).to(self.device)

        # 5. Loss
        self.loss_fn = HAMLLoss(
            mu_sep=self.mu_sep,
            mu_dyn=self.mu_dyn
        ).to(self.device)

    def fit(self, X, y):
        """
        Entraîne le modèle HAML.

        Args:
            X (np.ndarray): Données (n_samples, n_features)
            y (np.ndarray): Labels (n_samples,)

        Returns:
            self
        """
        # Normalisation
        X = self.scaler.fit_transform(X)

        # Initialisation
        self._initialize_architecture(X, y)

        print(f"HAML initialized: {self.n_levels} levels, {self.n_attractors_per_class} attractors/class")
        print(f"Coupling: alpha_bu={self.alpha_bu:.2f}, alpha_td={self.alpha_td:.2f}")

        # Vérification conditions
        self.coupling.verify_coupling_condition_C2(self.levels)

        self.is_fitted_ = True

        # Entraînement optionnel
        if self.train_on_fit and self.n_epochs > 0:
            print("\nStarting gradient training...")
            self.train_model(X, y)

        return self

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle par gradient descent avec stratégie 3 phases.

        Args:
            X_train (np.ndarray): Données d'entraînement (normalisées)
            y_train (np.ndarray): Labels
            X_val (np.ndarray, optional): Données de validation
            y_val (np.ndarray, optional): Labels de validation

        Returns:
            dict: Historique d'entraînement
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not initialized. Call fit() first.")

        # Créer optimiseur
        optimizer = ConstrainedOptimizer(
            self.parameters(),
            lr=self.lr
        )

        # Créer trainer
        trainer = HAMLTrainer(
            model=self,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            phase_config=PhaseConfig(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                phase1_epochs=max(1, self.n_epochs // 3),
                phase2_epochs=max(1, self.n_epochs // 3),
                td_warmup_power=1.0,
            ),
            device=self.device,
            verbose=True
        )

        # Entraîner
        history = trainer.train(X_train, y_train, X_val, y_val)

        return history

    def forward(self, X):
        """
        Intègre le système jusqu'à convergence et retourne les états finaux.

        Args:
            X (torch.Tensor): Données (batch, n_features)

        Returns:
            tuple: (final_states, n_steps, converged)
        """
        # Conditions initiales : projection sur tous les niveaux
        initial_states = self.spaces(X)

        # Intégration ODE
        final_states, n_steps, converged, _ = self.integrator.integrate(initial_states)

        return final_states, n_steps, converged

    def predict(self, X):
        """
        Prédit les classes.

        Args:
            X (np.ndarray): Données (n_samples, n_features)

        Returns:
            np.ndarray: Prédictions (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Normalisation
        X = self.scaler.transform(X)
        X_torch = torch.from_numpy(X).float().to(self.device)

        self.eval()
        with torch.no_grad():
            final_states, _, _ = self.forward(X_torch)
            scores = self._compute_hierarchical_scores(final_states)

            # Classe avec score max
            predictions = torch.argmax(scores, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Prédit les probabilités de classe.

        Args:
            X (np.ndarray): Données (n_samples, n_features)

        Returns:
            np.ndarray: Probabilités (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self.scaler.transform(X)
        X_torch = torch.from_numpy(X).float().to(self.device)

        self.eval()
        with torch.no_grad():
            final_states, _, _ = self.forward(X_torch)
            scores = self._compute_hierarchical_scores(final_states)

            # Softmax pour probabilités
            probs = torch.softmax(scores, dim=1)

        return probs.cpu().numpy()

    def score(self, X, y):
        """
        Calcule l'accuracy (interface scikit-learn).

        Args:
            X (np.ndarray): Données
            y (np.ndarray): Labels vrais

        Returns:
            float: Accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _level_weight(self, level_idx):
        """Retourne le poids d'agrégation d'un niveau hiérarchique."""
        if self.level_score_weighting == 'uniform':
            return 1.0
        if self.level_score_weighting == 'exponential':
            return float(2 ** level_idx)
        raise ValueError(
            f"Unknown level_score_weighting='{self.level_score_weighting}'. "
            "Use 'uniform' or 'exponential'."
        )

    def _compute_hierarchical_scores(self, final_states):
        """Agrège les scores classe par classe sur tous les niveaux."""
        batch_size = final_states[0].shape[0]
        scores = torch.zeros(batch_size, self.n_classes_, device=self.device)
        for l, (level, state) in enumerate(zip(self.levels, final_states)):
            level_scores = level.predict_class_scores(state)
            scores += self._level_weight(l) * level_scores
        return scores

    def get_final_states(self, X):
        """
        Retourne les états finaux après convergence (pour visualisation).

        Args:
            X (np.ndarray): Données

        Returns:
            list[np.ndarray]: États finaux à chaque niveau
        """
        X = self.scaler.transform(X)
        X_torch = torch.from_numpy(X).float().to(self.device)

        self.eval()
        with torch.no_grad():
            final_states, _, _ = self.forward(X_torch)

        return [s.cpu().numpy() for s in final_states]

    def diagnostics(self, X, y=None):
        """
        Retourne des diagnostics d'analyse après entraînement.

        Args:
            X (np.ndarray): Données d'évaluation
            y (np.ndarray, optional): Labels (pour accuracy par niveau)

        Returns:
            dict: {'force_stats': ..., 'level_accuracy': ...}
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        results = {
            'force_stats': compute_attraction_force_stats(self, X),
            'level_accuracy': None
        }

        if y is not None:
            X_norm = self.scaler.transform(X)
            results['level_accuracy'] = compute_level_accuracy(self, X_norm, y)

        return results

    def get_params(self, deep=True):
        """Paramètres pour scikit-learn GridSearch."""
        return {
            'n_levels': self.n_levels,
            'n_attractors_per_class': self.n_attractors_per_class,
            'alpha_bu': self.alpha_bu,
            'alpha_td': self.alpha_td,
            'gamma': self.gamma,
            'lambda_repulsion': self.lambda_repulsion,
            'rho_sigma_ratio': self.rho_sigma_ratio,
            'mu_sep': self.mu_sep,
            'mu_dyn': self.mu_dyn,
            'level_score_weighting': self.level_score_weighting,
            'device': self.device
        }

    def set_params(self, **params):
        """Mise à jour des paramètres (scikit-learn)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
