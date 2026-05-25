"""
Module HAML principal.

Orchestrateur du systÃ¨me hiÃ©rarchique d'attracteurs.
Interface compatible scikit-learn (fit/predict).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from ..projection import HierarchicalSpaces
from ..dynamics import Level, LevelVectorized, BidirectionalCoupling, ODEIntegrator
from ..training import (
    HAMLLoss,
    ConstrainedOptimizer,
    PhaseConfig,
    compute_level_accuracy,
    compute_attraction_force_stats
)
from ..training.trainer import HAMLTrainer


def _resolve_device(device):
    """Resolve runtime device with safe fallback."""
    if device is None:
        device = "auto"
    d = str(device).lower().strip()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        print("Requested device='cuda' but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    return d


class HAML(nn.Module, BaseEstimator, ClassifierMixin):
    """
    Hierarchical Attractor Machine Learning.

    SystÃ¨me de classification basÃ© sur une dynamique hiÃ©rarchique bidirectionnelle
    oÃ¹ la prÃ©diction Ã©merge de la convergence vers un bassin d'attraction stable.

    Architecture :
        X = H_0 â†’ H_1 â†’ ... â†’ H_L (projections PCA)
        Attracteurs {Î¼_{c,m}^(l)} Ã  chaque niveau
        Couplage bidirectionnel (bottom-up + top-down)
        IntÃ©gration ODE jusqu'Ã  rÃ©gime stationnaire
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
        sigma_init_mode='sqrt_d_std',
        mu_sep=0.1,
        mu_dyn=0.01,
        integrator_method='rk4',
        dt=0.1,
        max_steps=100,
        tol=1e-4,
        convergence_check_every=5,
        learn_projections=None,
        learn_alphas=False,
        lr=0.01,
        n_epochs=50,
        batch_size=32,
        train_on_fit=False,
        training_preset=None,
        phase1_max_steps=None,
        phase2_max_steps=None,
        phase3_max_steps=None,
        phase1_integrator_method=None,
        phase2_integrator_method=None,
        phase3_integrator_method=None,
        diagnostics_every_epochs=1,
        diagnostics_subset_size=None,
        level_score_weighting='exponential',
        use_vectorized_levels=False,
        repulsion_mode='global',
        device='auto'
    ):
        """
        Args:
            n_levels (int): Nombre de niveaux L
            level_dims (list, optional): Dimensions [d_1, ..., d_L] (auto si None)
            n_attractors_per_class (int): Nombre M d'attracteurs par classe
            alpha_bu (float): Coefficient bottom-up
            alpha_td (float): Coefficient top-down
            gamma (float): Amortissement
            lambda_repulsion (float): Ratio attraction/rÃ©pulsion Î»
            rho_sigma_ratio (float): Ratio d'initialisation rho/sigma (>1.0)
            mu_sep (float): Poids L_sep
            mu_dyn (float): Poids L_dyn
            integrator_method (str): 'euler', 'rk4', ou 'adjoint'
            dt (float): Pas de temps
            max_steps (int): Max itÃ©rations ODE
            tol (float): TolÃ©rance convergence
            learn_projections (bool | None): Affiner projections PCA.
                Si None, peut Ãªtre dÃ©cidÃ© par le preset d'entraÃ®nement.
            learn_alphas (bool): Apprendre Î±_bu, Î±_td
            level_score_weighting (str): 'uniform', 'exponential' ou 'learned_softmax'
            device (str): 'auto', 'cpu' ou 'cuda'
        """
        super().__init__()

        self.n_levels = n_levels
        self.level_dims = level_dims
        self.n_attractors_per_class = n_attractors_per_class
        self.device = _resolve_device(device)

        # HyperparamÃ¨tres stockÃ©s (pour scikit-learn)
        self.alpha_bu = alpha_bu
        self.alpha_td = alpha_td
        self.gamma = gamma
        self.lambda_repulsion = lambda_repulsion
        self.rho_sigma_ratio = rho_sigma_ratio
        self.sigma_init_mode = sigma_init_mode
        self.mu_sep = mu_sep
        self.mu_dyn = mu_dyn
        self.integrator_method = integrator_method
        self.dt = dt
        self.max_steps = max_steps
        self.tol = tol
        self.convergence_check_every = convergence_check_every
        self.learn_projections = learn_projections
        self.learn_alphas = learn_alphas
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_on_fit = train_on_fit
        self.training_preset = training_preset
        self.phase1_max_steps = phase1_max_steps
        self.phase2_max_steps = phase2_max_steps
        self.phase3_max_steps = phase3_max_steps
        self.phase1_integrator_method = phase1_integrator_method
        self.phase2_integrator_method = phase2_integrator_method
        self.phase3_integrator_method = phase3_integrator_method
        self.diagnostics_every_epochs = diagnostics_every_epochs
        self.diagnostics_subset_size = diagnostics_subset_size
        self.level_score_weighting = level_score_weighting
        self.use_vectorized_levels = use_vectorized_levels
        self.repulsion_mode = repulsion_mode
        self.level_weight_logits = nn.Parameter(torch.zeros(self.n_levels))
        self._apply_training_preset()
        if self.learn_projections is None:
            self.learn_projections = False

        # Modules (initialisÃ©s dans fit)
        self.scaler = StandardScaler()
        self.spaces = None
        self.levels = None
        self.coupling = None
        self.integrator = None
        self.loss_fn = None

        self.n_classes_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def _apply_training_preset(self):
        """Apply optional runtime preset while preserving explicit user overrides."""
        if self.training_preset is None:
            return
        preset = str(self.training_preset).strip().lower()
        if preset == "fast_train_cpu":
            if self.learn_projections is None:
                self.learn_projections = True
            if self.phase1_max_steps is None:
                self.phase1_max_steps = 20
            if self.phase2_max_steps is None:
                self.phase2_max_steps = 40
            if self.phase3_max_steps is None:
                self.phase3_max_steps = 100
            if self.phase1_integrator_method is None:
                self.phase1_integrator_method = "euler"
            if self.phase2_integrator_method is None:
                self.phase2_integrator_method = "euler"
            if self.phase3_integrator_method is None:
                self.phase3_integrator_method = "rk4"
            if self.diagnostics_every_epochs == 1:
                self.diagnostics_every_epochs = 1
            return
        if preset == "ultra_fast_train_cpu":
            if self.learn_projections is None:
                self.learn_projections = True
            if self.phase1_max_steps is None:
                self.phase1_max_steps = 20
            if self.phase2_max_steps is None:
                self.phase2_max_steps = 40
            if self.phase3_max_steps is None:
                self.phase3_max_steps = 100
            if self.phase1_integrator_method is None:
                self.phase1_integrator_method = "euler"
            if self.phase2_integrator_method is None:
                self.phase2_integrator_method = "euler"
            if self.phase3_integrator_method is None:
                self.phase3_integrator_method = "euler"
            if self.diagnostics_every_epochs == 1:
                self.diagnostics_every_epochs = 1
            return
        if preset == "fashion_cpu_accuracy":
            self.sigma_init_mode = "median_pairwise"
            if self.learn_projections is None:
                self.learn_projections = True
            if self.phase1_max_steps is None:
                self.phase1_max_steps = 20
            if self.phase2_max_steps is None:
                self.phase2_max_steps = 40
            if self.phase3_max_steps is None:
                self.phase3_max_steps = 100
            if self.phase1_integrator_method is None:
                self.phase1_integrator_method = "euler"
            if self.phase2_integrator_method is None:
                self.phase2_integrator_method = "euler"
            if self.phase3_integrator_method is None:
                self.phase3_integrator_method = "euler"
            if self.diagnostics_every_epochs == 1:
                self.diagnostics_every_epochs = 1
            return
        if preset == "fashion_cpu_runtime":
            self.sigma_init_mode = "sqrt_d_std"
            if self.learn_projections is None:
                self.learn_projections = True
            if self.phase1_max_steps is None:
                self.phase1_max_steps = 20
            if self.phase2_max_steps is None:
                self.phase2_max_steps = 40
            if self.phase3_max_steps is None:
                self.phase3_max_steps = 100
            if self.phase1_integrator_method is None:
                self.phase1_integrator_method = "euler"
            if self.phase2_integrator_method is None:
                self.phase2_integrator_method = "euler"
            if self.phase3_integrator_method is None:
                self.phase3_integrator_method = "euler"
            if self.diagnostics_every_epochs == 1:
                self.diagnostics_every_epochs = 1
            return
        raise ValueError(
            f"Unknown training_preset='{self.training_preset}'. "
            "Supported values: None, 'fast_train_cpu', 'ultra_fast_train_cpu', "
            "'fashion_cpu_accuracy', 'fashion_cpu_runtime'."
        )

    def _initialize_architecture(self, X, y):
        """
        Initialise la hiÃ©rarchie aprÃ¨s avoir vu les donnÃ©es.

        Args:
            X (np.ndarray): DonnÃ©es (n_samples, n_features)
            y (np.ndarray): Labels (n_samples,)
        """
        n_samples, input_dim = X.shape
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        # Dimensions par niveau (auto si non spÃ©cifiÃ©)
        if self.level_dims is None:
            # DÃ©croissance gÃ©omÃ©trique
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

        # 2. Niveaux hiÃ©rarchiques
        all_dims = [input_dim] + self.level_dims
        self.levels = nn.ModuleList()

        for l, dim in enumerate(all_dims):
            if self.use_vectorized_levels:
                level = LevelVectorized(
                    level_idx=l,
                    dim=dim,
                    n_classes=self.n_classes_,
                    n_attractors_per_class=self.n_attractors_per_class,
                    lambda_repulsion=self.lambda_repulsion,
                    rho_sigma_ratio=self.rho_sigma_ratio,
                    repulsion_mode=self.repulsion_mode,
                    sigma_init_mode=self.sigma_init_mode,
                ).to(self.device)
            else:
                level = Level(
                    level_idx=l,
                    dim=dim,
                    n_classes=self.n_classes_,
                    n_attractors_per_class=self.n_attractors_per_class,
                    lambda_repulsion=self.lambda_repulsion,
                    rho_sigma_ratio=self.rho_sigma_ratio,
                    sigma_init_mode=self.sigma_init_mode,
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

        # 4. IntÃ©grateur ODE
        self.integrator = ODEIntegrator(
            levels=list(self.levels),
            coupling=self.coupling,
            gamma=self.gamma,
            method=self.integrator_method,
            dt=self.dt,
            max_steps=self.max_steps,
            tol=self.tol,
            convergence_check_every=self.convergence_check_every,
        ).to(self.device)

        # 5. Loss
        self.loss_fn = HAMLLoss(
            mu_sep=self.mu_sep,
            mu_dyn=self.mu_dyn
        ).to(self.device)

    def fit(self, X, y):
        """
        EntraÃ®ne le modÃ¨le HAML.

        Args:
            X (np.ndarray): DonnÃ©es (n_samples, n_features)
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

        # VÃ©rification conditions
        self.coupling.verify_coupling_condition_C2(self.levels)

        self.is_fitted_ = True

        # EntraÃ®nement optionnel
        if self.train_on_fit and self.n_epochs > 0:
            print("\nStarting gradient training...")
            self.train_model(X, y)

        return self

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        EntraÃ®ne le modÃ¨le par gradient descent avec stratÃ©gie 3 phases.

        Args:
            X_train (np.ndarray): DonnÃ©es d'entraÃ®nement (normalisÃ©es)
            y_train (np.ndarray): Labels
            X_val (np.ndarray, optional): DonnÃ©es de validation
            y_val (np.ndarray, optional): Labels de validation

        Returns:
            dict: Historique d'entraÃ®nement
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not initialized. Call fit() first.")

        # CrÃ©er optimiseur
        optimizer = ConstrainedOptimizer(
            self.parameters(),
            lr=self.lr
        )

        # CrÃ©er trainer
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
                phase1_max_steps=self.phase1_max_steps,
                phase2_max_steps=self.phase2_max_steps,
                phase3_max_steps=self.phase3_max_steps,
                phase1_integrator_method=self.phase1_integrator_method,
                phase2_integrator_method=self.phase2_integrator_method,
                phase3_integrator_method=self.phase3_integrator_method,
                diagnostics_every_epochs=self.diagnostics_every_epochs,
                diagnostics_subset_size=self.diagnostics_subset_size,
            ),
            device=self.device,
            verbose=True
        )

        # EntraÃ®ner
        history = trainer.train(X_train, y_train, X_val, y_val)

        return history

    def forward(self, X):
        """
        IntÃ¨gre le systÃ¨me jusqu'Ã  convergence et retourne les Ã©tats finaux.

        Args:
            X (torch.Tensor): DonnÃ©es (batch, n_features)

        Returns:
            tuple: (final_states, n_steps, converged)
        """
        # Conditions initiales : projection sur tous les niveaux
        initial_states = self.spaces(X)

        # IntÃ©gration ODE
        final_states, n_steps, converged, _ = self.integrator.integrate(initial_states)

        return final_states, n_steps, converged

    def predict(self, X):
        """
        PrÃ©dit les classes.

        Args:
            X (np.ndarray): DonnÃ©es (n_samples, n_features)

        Returns:
            np.ndarray: PrÃ©dictions (n_samples,)
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
        PrÃ©dit les probabilitÃ©s de classe.

        Args:
            X (np.ndarray): DonnÃ©es (n_samples, n_features)

        Returns:
            np.ndarray: ProbabilitÃ©s (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self.scaler.transform(X)
        X_torch = torch.from_numpy(X).float().to(self.device)

        self.eval()
        with torch.no_grad():
            final_states, _, _ = self.forward(X_torch)
            scores = self._compute_hierarchical_scores(final_states)

            # Softmax pour probabilitÃ©s
            probs = torch.softmax(scores, dim=1)

        return probs.cpu().numpy()

    def score(self, X, y):
        """
        Calcule l'accuracy (interface scikit-learn).

        Args:
            X (np.ndarray): DonnÃ©es
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
        if self.level_score_weighting == 'learned_softmax':
            weights = torch.softmax(self.level_weight_logits, dim=0)
            return float(weights[level_idx].detach().item())
        raise ValueError(
            "Unknown level_score_weighting='{}'. Use 'uniform', 'exponential' or 'learned_softmax'.".format(self.level_score_weighting)
        )

    def _get_level_weights_tensor(self, device):
        """Retourne les poids d'agrégation sous forme de tenseur."""
        if self.level_score_weighting == 'uniform':
            return torch.ones(self.n_levels, device=device, dtype=torch.float32)
        if self.level_score_weighting == 'exponential':
            return torch.tensor([2 ** l for l in range(self.n_levels)], device=device, dtype=torch.float32)
        if self.level_score_weighting == 'learned_softmax':
            return torch.softmax(self.level_weight_logits.to(device), dim=0)
        raise ValueError(
            "Unknown level_score_weighting='{}'. Use 'uniform', 'exponential' or 'learned_softmax'.".format(self.level_score_weighting)
        )

    def _compute_hierarchical_scores(self, final_states):
        """Agrège les scores classe par classe sur tous les niveaux."""
        batch_size = final_states[0].shape[0]
        scores = torch.zeros(batch_size, self.n_classes_, device=self.device)
        level_weights = self._get_level_weights_tensor(final_states[0].device)
        for l, (level, state) in enumerate(zip(self.levels, final_states)):
            level_scores = level.predict_class_scores(state)
            scores += level_weights[l] * level_scores
        return scores

    def get_final_states(self, X):
        """
        Retourne les Ã©tats finaux aprÃ¨s convergence (pour visualisation).

        Args:
            X (np.ndarray): DonnÃ©es

        Returns:
            list[np.ndarray]: Ã‰tats finaux Ã  chaque niveau
        """
        X = self.scaler.transform(X)
        X_torch = torch.from_numpy(X).float().to(self.device)

        self.eval()
        with torch.no_grad():
            final_states, _, _ = self.forward(X_torch)

        return [s.cpu().numpy() for s in final_states]

    def diagnostics(self, X, y=None):
        """
        Retourne des diagnostics d'analyse aprÃ¨s entraÃ®nement.

        Args:
            X (np.ndarray): DonnÃ©es d'Ã©valuation
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
        """ParamÃ¨tres pour scikit-learn GridSearch."""
        return {
            'n_levels': self.n_levels,
            'n_attractors_per_class': self.n_attractors_per_class,
            'alpha_bu': self.alpha_bu,
            'alpha_td': self.alpha_td,
            'gamma': self.gamma,
            'lambda_repulsion': self.lambda_repulsion,
            'rho_sigma_ratio': self.rho_sigma_ratio,
            'sigma_init_mode': self.sigma_init_mode,
            'learn_projections': self.learn_projections,
            'mu_sep': self.mu_sep,
            'mu_dyn': self.mu_dyn,
            'convergence_check_every': self.convergence_check_every,
            'training_preset': self.training_preset,
            'phase1_max_steps': self.phase1_max_steps,
            'phase2_max_steps': self.phase2_max_steps,
            'phase3_max_steps': self.phase3_max_steps,
            'phase1_integrator_method': self.phase1_integrator_method,
            'phase2_integrator_method': self.phase2_integrator_method,
            'phase3_integrator_method': self.phase3_integrator_method,
            'diagnostics_every_epochs': self.diagnostics_every_epochs,
            'diagnostics_subset_size': self.diagnostics_subset_size,
            'level_score_weighting': self.level_score_weighting,
            'use_vectorized_levels': self.use_vectorized_levels,
            'repulsion_mode': self.repulsion_mode,
            'device': self.device
        }

    def set_params(self, **params):
        """Mise Ã  jour des paramÃ¨tres (scikit-learn)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
