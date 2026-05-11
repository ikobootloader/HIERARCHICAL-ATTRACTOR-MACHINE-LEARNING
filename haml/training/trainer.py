"""
Module Trainer.

Orchestration de l'entraînement HAML avec :
- Backprop à travers trajectoires ODE
- Stratégie de couplage progressif (3 phases)
- Monitoring de convergence
"""

import torch
import numpy as np
from tqdm import tqdm
from .metrics import compute_level_accuracy


class HAMLTrainer:
    """
    Entraîneur HAML avec backprop à travers trajectoires ODE.

    Stratégie 3 phases :
    - Phase 1 : Niveaux indépendants (α=0), stabilise attracteurs
    - Phase 2 : Couplage progressif, synchronise niveaux
    - Phase 3 : Entraînement conjoint tous paramètres
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        n_epochs=50,
        batch_size=32,
        phase1_epochs=15,
        phase2_epochs=15,
        td_warmup_power=1.0,
        level_divergence_threshold=0.15,
        divergence_patience=2,
        lr_decay_on_divergence=0.5,
        min_lr=1e-4,
        early_stop_on_divergence=True,
        adaptive_mu_sep=True,
        adaptive_mu_sep_phase3_only=True,
        mu_sep_trigger_divergence=0.05,
        mu_sep_patience=1,
        mu_sep_growth_factor=1.2,
        mu_sep_max=1.0,
        soft_landing_epoch=None,
        soft_landing_trigger_divergence=None,
        soft_landing_lr_factor=0.1,
        soft_landing_freeze_mu=True,
        collapse_guard_enabled=False,
        collapse_guard_start_epoch=18,
        collapse_guard_delta_div_threshold=0.10,
        collapse_guard_mu_sep_boost=1.5,
        collapse_guard_lr_factor=0.3,
        device='cpu',
        verbose=True
    ):
        """
        Args:
            model (HAML): Modèle HAML
            optimizer (ConstrainedOptimizer): Optimiseur avec contraintes
            loss_fn (HAMLLoss): Fonction de loss
            n_epochs (int): Nombre total d'epochs
            batch_size (int): Taille des batchs
            phase1_epochs (int): Epochs phase 1 (niveaux indépendants)
            phase2_epochs (int): Epochs phase 2 (couplage progressif)
            td_warmup_power (float): Exposant de warmup top-down en phase 2
            level_divergence_threshold (float): Seuil max(level_acc)-min(level_acc)
            divergence_patience (int): Nb d'epochs divergentes tolérées avant stop
            lr_decay_on_divergence (float): Facteur multiplicatif de LR en divergence
            min_lr (float): LR minimal
            early_stop_on_divergence (bool): Active arrêt anticipé défensif
            adaptive_mu_sep (bool): Active montée adaptative de mu_sep
            adaptive_mu_sep_phase3_only (bool): N'active mu_sep adaptatif qu'en phase 3
            mu_sep_trigger_divergence (float): Seuil préventif de divergence
            mu_sep_patience (int): Nb d'epochs consécutives au-dessus du seuil avant hausse de mu_sep
            mu_sep_growth_factor (float): Multiplicateur de mu_sep
            mu_sep_max (float): Cap supérieur de mu_sep
            soft_landing_epoch (int|None): Epoch (1-based) déclencheur soft landing
            soft_landing_trigger_divergence (float|None): Déclenchement soft landing si level_div dépasse ce seuil
            soft_landing_lr_factor (float): Facteur LR au soft landing
            soft_landing_freeze_mu (bool): Gèle les positions mu au soft landing
            collapse_guard_enabled (bool): Active la détection de collapse brutal inter-niveaux
            collapse_guard_start_epoch (int): Epoch minimale (1-based) avant activation du guard
            collapse_guard_delta_div_threshold (float): Seuil de saut sur delta(level_div)
            collapse_guard_mu_sep_boost (float): Multiplicateur ponctuel de mu_sep
            collapse_guard_lr_factor (float): Facteur multiplicatif de LR lors du collapse
            device (str): Device
            verbose (bool): Affichage
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.td_warmup_power = td_warmup_power
        self.level_divergence_threshold = level_divergence_threshold
        self.divergence_patience = divergence_patience
        self.lr_decay_on_divergence = lr_decay_on_divergence
        self.min_lr = min_lr
        self.early_stop_on_divergence = early_stop_on_divergence
        self.adaptive_mu_sep = adaptive_mu_sep
        self.adaptive_mu_sep_phase3_only = adaptive_mu_sep_phase3_only
        self.mu_sep_trigger_divergence = mu_sep_trigger_divergence
        self.mu_sep_patience = mu_sep_patience
        self.mu_sep_growth_factor = mu_sep_growth_factor
        self.mu_sep_max = mu_sep_max
        self.soft_landing_epoch = soft_landing_epoch
        self.soft_landing_trigger_divergence = soft_landing_trigger_divergence
        self.soft_landing_lr_factor = soft_landing_lr_factor
        self.soft_landing_freeze_mu = soft_landing_freeze_mu
        self.collapse_guard_enabled = collapse_guard_enabled
        self.collapse_guard_start_epoch = collapse_guard_start_epoch
        self.collapse_guard_delta_div_threshold = collapse_guard_delta_div_threshold
        self.collapse_guard_mu_sep_boost = collapse_guard_mu_sep_boost
        self.collapse_guard_lr_factor = collapse_guard_lr_factor
        self.device = device
        self.verbose = verbose

        # Historique
        self.history = {
            'loss': [],
            'loss_ce': [],
            'loss_sep': [],
            'loss_dyn': [],
            'accuracy': [],
            'val_accuracy': [],
            'level_accuracy': [],
            'level_divergence': [],
            'lr': [],
            'mu_sep': []
        }

        # Sauvegarde des alphas originaux
        self.alpha_bu_target = model.coupling.alpha_bu.item()
        self.alpha_td_target = model.coupling.alpha_td.item()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle avec stratégie 3 phases.

        Args:
            X_train (np.ndarray): Données d'entraînement
            y_train (np.ndarray): Labels d'entraînement
            X_val (np.ndarray, optional): Données de validation
            y_val (np.ndarray, optional): Labels de validation

        Returns:
            dict: Historique d'entraînement
        """
        # Normalisation déjà faite dans fit()
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)

        n_samples = len(X_train)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        if self.verbose:
            print("\n" + "="*70)
            print("HAML Training - 3 Phase Strategy")
            print("="*70)
            print(f"Total epochs: {self.n_epochs}")
            print(f"  Phase 1 (independent): epochs 0-{self.phase1_epochs}")
            print(f"  Phase 2 (progressive): epochs {self.phase1_epochs}-{self.phase1_epochs + self.phase2_epochs}")
            print(f"  Phase 3 (joint): epochs {self.phase1_epochs + self.phase2_epochs}-{self.n_epochs}")
            print("="*70 + "\n")

        # Entraînement
        divergence_streak = 0
        mu_sep_streak = 0
        soft_landing_applied = False
        collapse_guard_applied = False
        prev_level_div = None
        for epoch in range(self.n_epochs):
            # Détermination de la phase
            if epoch < self.phase1_epochs:
                phase = 1
                self._set_coupling_phase1()
            elif epoch < self.phase1_epochs + self.phase2_epochs:
                phase = 2
                progress = (epoch - self.phase1_epochs) / self.phase2_epochs
                self._set_coupling_phase2(progress)
            else:
                phase = 3
                self._set_coupling_phase3()

            # Epoch
            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_sep = 0.0
            epoch_loss_dyn = 0.0

            # Shuffle
            indices = torch.randperm(n_samples)

            if self.verbose:
                pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.n_epochs} [Phase {phase}]")
            else:
                pbar = range(n_batches)

            for batch_idx in pbar:
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]

                # Forward
                self.optimizer.zero_grad()

                # Intégration ODE avec trajectoire
                initial_states = self.model.spaces(X_batch)
                final_states, _, _, trajectory = self.model.integrator.integrate(
                    initial_states, return_trajectory=True
                )

                # Loss
                loss_dict = self.loss_fn(
                    final_states,
                    list(self.model.levels),
                    y_batch,
                    trajectory=trajectory if self.loss_fn.mu_dyn > 0 else None
                )

                loss = loss_dict['total']

                # Backward
                loss.backward()

                # Step avec contraintes
                self.optimizer.step(self.model)

                # Stats
                epoch_loss += loss.item()
                epoch_loss_ce += loss_dict['ce']
                epoch_loss_sep += loss_dict['sep']
                epoch_loss_dyn += loss_dict['dyn']

                if self.verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f"{loss.item():.3f}",
                        'ce': f"{loss_dict['ce']:.3f}"
                    })

            # Moyennes
            epoch_loss /= n_batches
            epoch_loss_ce /= n_batches
            epoch_loss_sep /= n_batches
            epoch_loss_dyn /= n_batches

            # Accuracy
            with torch.no_grad():
                y_pred = self.model.predict(X_train)
                accuracy = np.mean(y_pred == y_train)
                level_accuracy = compute_level_accuracy(self.model, X_train, y_train)

            # Validation
            if X_val is not None:
                with torch.no_grad():
                    y_pred_val = self.model.predict(X_val)
                    val_accuracy = np.mean(y_pred_val == y_val)
            else:
                val_accuracy = None

            # Historique
            self.history['loss'].append(epoch_loss)
            self.history['loss_ce'].append(epoch_loss_ce)
            self.history['loss_sep'].append(epoch_loss_sep)
            self.history['loss_dyn'].append(epoch_loss_dyn)
            self.history['accuracy'].append(accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['level_accuracy'].append(level_accuracy)
            level_div = max(level_accuracy) - min(level_accuracy)
            self.history['level_divergence'].append(level_div)
            self.history['lr'].append(self.optimizer.get_lr())
            self.history['mu_sep'].append(self.loss_fn.mu_sep)

            if (
                self.collapse_guard_enabled
                and not collapse_guard_applied
                and prev_level_div is not None
                and (epoch + 1) >= self.collapse_guard_start_epoch
            ):
                delta_div = level_div - prev_level_div
                if delta_div > self.collapse_guard_delta_div_threshold:
                    new_mu_sep = min(self.mu_sep_max, self.loss_fn.mu_sep * self.collapse_guard_mu_sep_boost)
                    if new_mu_sep > self.loss_fn.mu_sep:
                        self.loss_fn.set_weights(mu_sep=new_mu_sep)
                    new_lr = max(self.min_lr, self.optimizer.get_lr() * self.collapse_guard_lr_factor)
                    self.optimizer.set_lr(new_lr)
                    collapse_guard_applied = True
                    if self.verbose:
                        print(
                            f"[collapse-guard] epoch={epoch+1}, delta_div={delta_div:.3f} > "
                            f"{self.collapse_guard_delta_div_threshold:.3f}; "
                            f"mu_sep -> {self.loss_fn.mu_sep:.4f}, lr -> {new_lr:.6f}"
                        )

            use_adaptive_mu_sep = self.adaptive_mu_sep
            if self.adaptive_mu_sep_phase3_only and phase != 3:
                use_adaptive_mu_sep = False

            if use_adaptive_mu_sep and level_div > self.mu_sep_trigger_divergence:
                mu_sep_streak += 1
            else:
                mu_sep_streak = 0

            if use_adaptive_mu_sep and mu_sep_streak >= self.mu_sep_patience:
                new_mu_sep = min(self.mu_sep_max, self.loss_fn.mu_sep * self.mu_sep_growth_factor)
                if new_mu_sep > self.loss_fn.mu_sep:
                    self.loss_fn.set_weights(mu_sep=new_mu_sep)
                    if self.verbose:
                        print(
                            f"[mu-sep] level_div={level_div:.3f} > "
                            f"{self.mu_sep_trigger_divergence:.3f} "
                            f"(streak={mu_sep_streak}); mu_sep -> {new_mu_sep:.4f}"
                        )

            # Soft landing: mode préventif (epoch) ou réactif (divergence), une seule fois.
            soft_landing_by_epoch = (
                self.soft_landing_epoch is not None and (epoch + 1) >= self.soft_landing_epoch
            )
            soft_landing_by_div = (
                self.soft_landing_trigger_divergence is not None
                and level_div >= self.soft_landing_trigger_divergence
            )
            if not soft_landing_applied and (soft_landing_by_epoch or soft_landing_by_div):
                new_lr = max(self.min_lr, self.optimizer.get_lr() * self.soft_landing_lr_factor)
                self.optimizer.set_lr(new_lr)
                if self.soft_landing_freeze_mu:
                    self._freeze_mu_positions()
                soft_landing_applied = True
                if self.verbose:
                    reason = (
                        f"epoch>={self.soft_landing_epoch}"
                        if soft_landing_by_epoch
                        else f"level_div>={self.soft_landing_trigger_divergence:.3f}"
                    )
                    print(
                        f"[soft-landing] reason={reason}, epoch={epoch+1}, "
                        f"lr -> {new_lr:.6f}, freeze_mu={self.soft_landing_freeze_mu}"
                    )

            if level_div > self.level_divergence_threshold:
                divergence_streak += 1
                new_lr = max(self.min_lr, self.optimizer.get_lr() * self.lr_decay_on_divergence)
                self.optimizer.set_lr(new_lr)
                if self.verbose:
                    print(
                        f"[stability] level_div={level_div:.3f} > "
                        f"{self.level_divergence_threshold:.3f}; lr -> {new_lr:.6f}"
                    )
            else:
                divergence_streak = 0

            # Affichage
            if self.verbose:
                msg = f"Epoch {epoch+1}: loss={epoch_loss:.3f}, acc={accuracy:.3f}"
                msg += f", level_acc={['{:.3f}'.format(a) for a in level_accuracy]}"
                msg += f", level_div={level_div:.3f}, lr={self.optimizer.get_lr():.6f}"
                if val_accuracy is not None:
                    msg += f", val_acc={val_accuracy:.3f}"
                print(msg)

            if self.early_stop_on_divergence and divergence_streak >= self.divergence_patience:
                if self.verbose:
                    print(
                        f"[early-stop] divergence streak={divergence_streak} "
                        f"(threshold={self.level_divergence_threshold:.3f})"
                    )
                break

            prev_level_div = level_div

        if self.verbose:
            print("\n" + "="*70)
            print("Training completed")
            print(f"Final accuracy: {self.history['accuracy'][-1]:.3f}")
            print("="*70)

        return self.history

    def _set_coupling_phase1(self):
        """Phase 1 : Niveaux indépendants (α_bu = α_td = 0)."""
        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(1e-6)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(1e-6)))

    def _set_coupling_phase2(self, progress):
        """
        Phase 2 : Couplage progressif (α croît linéairement).

        Args:
            progress (float): Progression 0->1
        """
        alpha_bu = progress * self.alpha_bu_target
        alpha_td = (progress ** self.td_warmup_power) * self.alpha_td_target

        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(alpha_bu + 1e-6)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(alpha_td + 1e-6)))

    def _set_coupling_phase3(self):
        """Phase 3 : Couplage complet (α = valeur cible)."""
        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(self.alpha_bu_target)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(self.alpha_td_target)))

    def _freeze_mu_positions(self):
        """Gèle les positions des attracteurs (mu) pour limiter la dérive tardive."""
        for level in self.model.levels:
            for c in range(level.n_classes):
                for attractor in level.attractors[str(c)]:
                    attractor.position.requires_grad_(False)

    def plot_history(self):
        """Visualise l'historique d'entraînement."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss totale
        axes[0, 0].plot(self.history['loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Loss composantes
        axes[0, 1].plot(self.history['loss_ce'], label='CE', linewidth=2)
        axes[0, 1].plot(self.history['loss_sep'], label='Sep', linewidth=2)
        axes[0, 1].plot(self.history['loss_dyn'], label='Dyn', linewidth=2)
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Accuracy
        axes[1, 0].plot(self.history['accuracy'], 'g-', linewidth=2)
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)

        # Phases
        axes[1, 1].axvspan(0, self.phase1_epochs, alpha=0.2, color='red', label='Phase 1')
        axes[1, 1].axvspan(self.phase1_epochs, self.phase1_epochs + self.phase2_epochs,
                          alpha=0.2, color='orange', label='Phase 2')
        axes[1, 1].axvspan(self.phase1_epochs + self.phase2_epochs, self.n_epochs,
                          alpha=0.2, color='green', label='Phase 3')
        axes[1, 1].plot(self.history['accuracy'], 'k-', linewidth=2)
        axes[1, 1].set_title('Training Phases')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
