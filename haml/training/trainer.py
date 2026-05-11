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
        self.device = device
        self.verbose = verbose

        # Historique
        self.history = {
            'loss': [],
            'loss_ce': [],
            'loss_sep': [],
            'loss_dyn': [],
            'accuracy': [],
            'val_accuracy': []
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

            # Affichage
            if self.verbose:
                msg = f"Epoch {epoch+1}: loss={epoch_loss:.3f}, acc={accuracy:.3f}"
                if val_accuracy is not None:
                    msg += f", val_acc={val_accuracy:.3f}"
                print(msg)

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
        alpha_td = progress * self.alpha_td_target

        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(alpha_bu + 1e-6)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(alpha_td + 1e-6)))

    def _set_coupling_phase3(self):
        """Phase 3 : Couplage complet (α = valeur cible)."""
        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(self.alpha_bu_target)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(self.alpha_td_target)))

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
