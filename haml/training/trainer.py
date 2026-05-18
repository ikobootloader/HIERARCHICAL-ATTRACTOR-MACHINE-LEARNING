"""
Module Trainer.

Orchestration de l'entraÃ®nement HAML avec :
- Backprop Ã  travers trajectoires ODE
- StratÃ©gie de couplage progressif (3 phases)
- Monitoring de convergence
"""

import torch
import numpy as np
from tqdm import tqdm
from .metrics import compute_level_accuracy
from .config import (
    AdaptiveMuSepConfig,
    CollapseGuardConfig,
    LevelRecoveryConfig,
    PhaseConfig,
    SoftLandingConfig,
    StabilityConfig,
)


class HAMLTrainer:
    """
    EntraÃ®neur HAML avec backprop Ã  travers trajectoires ODE.

    StratÃ©gie 3 phases :
    - Phase 1 : Niveaux indÃ©pendants (Î±=0), stabilise attracteurs
    - Phase 2 : Couplage progressif, synchronise niveaux
    - Phase 3 : EntraÃ®nement conjoint tous paramÃ¨tres
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        phase_config=None,
        stability_config=None,
        adaptive_mu_sep_config=None,
        soft_landing_config=None,
        collapse_guard_config=None,
        level_recovery_config=None,
        device='cpu',
        verbose=True
    ):
        """
        Args:
            model (HAML): ModÃ¨le HAML
            optimizer (ConstrainedOptimizer): Optimiseur avec contraintes
            loss_fn (HAMLLoss): Fonction de loss
            phase_config (PhaseConfig|None): Configuration de phases
            stability_config (StabilityConfig|None): Configuration de stabilitÃ©
            adaptive_mu_sep_config (AdaptiveMuSepConfig|None): Configuration mu_sep adaptatif
            soft_landing_config (SoftLandingConfig|None): Configuration soft landing
            collapse_guard_config (CollapseGuardConfig|None): Configuration collapse guard
            level_recovery_config (LevelRecoveryConfig|None): Configuration de recuperation par niveau
            device (str): Device
            verbose (bool): Affichage
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.phase_config = phase_config or PhaseConfig()
        self.stability_config = stability_config or StabilityConfig()
        self.adaptive_mu_sep_config = adaptive_mu_sep_config or AdaptiveMuSepConfig()
        self.soft_landing_config = soft_landing_config or SoftLandingConfig()
        self.collapse_guard_config = collapse_guard_config or CollapseGuardConfig()
        self.level_recovery_config = level_recovery_config or LevelRecoveryConfig()
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
            'mu_sep': [],
            'level_attractor_diagnostics': [],
            'convergence_fraction': [],
            'phase_max_steps': [],
            'events': [],
        }

        # Sauvegarde des alphas originaux
        self.alpha_bu_target = model.coupling.alpha_bu.item()
        self.alpha_td_target = model.coupling.alpha_td.item()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        EntraÃ®ne le modÃ¨le avec stratÃ©gie 3 phases.

        Args:
            X_train (np.ndarray): DonnÃ©es d'entraÃ®nement
            y_train (np.ndarray): Labels d'entraÃ®nement
            X_val (np.ndarray, optional): DonnÃ©es de validation
            y_val (np.ndarray, optional): Labels de validation

        Returns:
            dict: Historique d'entraÃ®nement
        """
        # Normalisation dÃ©jÃ  faite dans fit()
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)

        n_samples = len(X_train)
        n_batches = (n_samples + self.phase_config.batch_size - 1) // self.phase_config.batch_size

        if self.verbose:
            print("\n" + "="*70)
            print("HAML Training - 3 Phase Strategy")
            print("="*70)
            print(f"Total epochs: {self.phase_config.n_epochs}")
            print(f"  Phase 1 (independent): epochs 0-{self.phase_config.phase1_epochs}")
            print(
                f"  Phase 2 (progressive): epochs {self.phase_config.phase1_epochs}-"
                f"{self.phase_config.phase1_epochs + self.phase_config.phase2_epochs}"
            )
            print(
                f"  Phase 3 (joint): epochs "
                f"{self.phase_config.phase1_epochs + self.phase_config.phase2_epochs}-"
                f"{self.phase_config.n_epochs}"
            )
            print("="*70 + "\n")

        # EntraÃ®nement
        divergence_streak = 0
        lr_decay_events = 0
        last_lr_decay_epoch = -10**9
        mu_sep_streak = 0
        level_recovery_streak = [0 for _ in range(len(self.model.levels))]
        level_recovery_triggers = 0
        level_recovery_td_cooldown = 0
        soft_landing_applied = False
        collapse_guard_applied = False
        prev_level_div = None
        for epoch in range(self.phase_config.n_epochs):
            recovery_triggered_this_epoch = False
            # DÃ©termination de la phase
            if epoch < self.phase_config.phase1_epochs:
                phase = 1
                self._set_coupling_phase1()
            elif epoch < self.phase_config.phase1_epochs + self.phase_config.phase2_epochs:
                phase = 2
                progress = (epoch - self.phase_config.phase1_epochs) / self.phase_config.phase2_epochs
                self._set_coupling_phase2(progress)
            else:
                phase = 3
                if level_recovery_td_cooldown > 0:
                    self._set_coupling_phase3(td_scale=self.level_recovery_config.td_scale_during_cooldown)
                    level_recovery_td_cooldown -= 1
                else:
                    self._set_coupling_phase3()
            self._set_phase_max_steps(phase)

            # Epoch
            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_sep = 0.0
            epoch_loss_dyn = 0.0
            epoch_convergence_fraction = 0.0
            epoch_convergence_count = 0

            # Shuffle
            indices = torch.randperm(n_samples)

            if self.verbose:
                pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.phase_config.n_epochs} [Phase {phase}]")
            else:
                pbar = range(n_batches)

            for batch_idx in pbar:
                start_idx = batch_idx * self.phase_config.batch_size
                end_idx = min(start_idx + self.phase_config.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]

                # Forward
                self.optimizer.zero_grad()

                # IntÃ©gration ODE avec trajectoire uniquement si nÃ©cessaire
                need_trajectory = self.loss_fn.mu_dyn > 0
                initial_states = self.model.spaces(X_batch)
                final_states, _, _, trajectory = self.model.integrator.integrate(
                    initial_states, return_trajectory=need_trajectory
                )
                if self.model.integrator.last_convergence_fraction is not None:
                    epoch_convergence_fraction += float(self.model.integrator.last_convergence_fraction)
                    epoch_convergence_count += 1

                # Loss
                loss_dict = self.loss_fn(
                    final_states,
                    list(self.model.levels),
                    y_batch,
                    trajectory=trajectory if need_trajectory else None
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
            mean_convergence_fraction = (
                epoch_convergence_fraction / epoch_convergence_count
                if epoch_convergence_count > 0 else None
            )

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
            self.history['convergence_fraction'].append(mean_convergence_fraction)
            self.history['phase_max_steps'].append(int(self.model.integrator.max_steps))
            self.history['level_attractor_diagnostics'].append(
                self._compute_level_attractor_diagnostics()
            )

            if (
                self.collapse_guard_config.enabled
                and not collapse_guard_applied
                and prev_level_div is not None
                and (epoch + 1) >= self.collapse_guard_config.start_epoch
            ):
                delta_div = level_div - prev_level_div
                if delta_div > self.collapse_guard_config.delta_div_threshold:
                    new_mu_sep = min(
                        self.adaptive_mu_sep_config.max_value,
                        self.loss_fn.mu_sep * self.collapse_guard_config.mu_sep_boost,
                    )
                    if new_mu_sep > self.loss_fn.mu_sep:
                        self.loss_fn.set_weights(mu_sep=new_mu_sep)
                    new_lr = max(
                        self.stability_config.min_lr,
                        self.optimizer.get_lr() * self.collapse_guard_config.lr_factor,
                    )
                    self.optimizer.set_lr(new_lr)
                    collapse_guard_applied = True
                    if self.verbose:
                        print(
                            f"[collapse-guard] epoch={epoch+1}, delta_div={delta_div:.3f} > "
                            f"{self.collapse_guard_config.delta_div_threshold:.3f}; "
                            f"mu_sep -> {self.loss_fn.mu_sep:.4f}, lr -> {new_lr:.6f}"
                        )

            use_adaptive_mu_sep = self.adaptive_mu_sep_config.enabled
            if self.adaptive_mu_sep_config.phase3_only and phase != 3:
                use_adaptive_mu_sep = False

            if use_adaptive_mu_sep and level_div > self.adaptive_mu_sep_config.trigger_divergence:
                mu_sep_streak += 1
            else:
                mu_sep_streak = 0

            if use_adaptive_mu_sep and mu_sep_streak >= self.adaptive_mu_sep_config.patience:
                new_mu_sep = min(
                    self.adaptive_mu_sep_config.max_value,
                    self.loss_fn.mu_sep * self.adaptive_mu_sep_config.growth_factor,
                )
                if new_mu_sep > self.loss_fn.mu_sep:
                    self.loss_fn.set_weights(mu_sep=new_mu_sep)
                    if self.verbose:
                        print(
                            f"[mu-sep] level_div={level_div:.3f} > "
                            f"{self.adaptive_mu_sep_config.trigger_divergence:.3f} "
                            f"(streak={mu_sep_streak}); mu_sep -> {new_mu_sep:.4f}"
                        )

            if (
                self.level_recovery_config.enabled
                and phase == 3
                and level_recovery_triggers < self.level_recovery_config.max_triggers
            ):
                # Keep streak accumulation independent from global gating conditions.
                # This tracks true "stuck near chance" duration, then gate only triggering.
                stuck_level_idx = self._detect_stuck_level(level_accuracy, level_recovery_streak)
                if (
                    stuck_level_idx is not None
                    and level_div >= self.level_recovery_config.require_divergence
                    and self._other_levels_are_strong(level_accuracy, stuck_level_idx)
                    and self._has_time_for_recovery(epoch)
                ):
                    lr_before_recovery = float(self.optimizer.get_lr())
                    self._apply_level_recovery(stuck_level_idx, X_train_t, y_train_t)
                    level_recovery_triggers += 1
                    level_recovery_td_cooldown = max(
                        0,
                        int(self.level_recovery_config.td_cooldown_epochs),
                    )

                    new_mu_sep = min(
                        self.adaptive_mu_sep_config.max_value,
                        self.loss_fn.mu_sep * self.level_recovery_config.mu_sep_boost,
                    )
                    if new_mu_sep > self.loss_fn.mu_sep:
                        self.loss_fn.set_weights(mu_sep=new_mu_sep)

                    new_lr = max(
                        self.stability_config.min_lr,
                        lr_before_recovery * self.level_recovery_config.lr_factor,
                    )
                    self.optimizer.set_lr(new_lr)
                    level_recovery_streak = [0 for _ in range(len(self.model.levels))]
                    recovery_triggered_this_epoch = True
                    self.history['events'].append(
                        {
                            'type': 'level_recovery',
                            'epoch': int(epoch + 1),
                            'level_idx': int(stuck_level_idx),
                            'level_accuracy': float(level_accuracy[stuck_level_idx]),
                            'level_divergence': float(level_div),
                            'lr_before': lr_before_recovery,
                            'lr_after': float(new_lr),
                            'mutex_prevented_stability_decay': bool(
                                self.stability_config.skip_lr_decay_if_recovery_triggered
                            ),
                        }
                    )
                    if self.verbose:
                        print(
                            f"[level-recovery] epoch={epoch+1}, level={stuck_level_idx}, "
                            f"level_acc={level_accuracy[stuck_level_idx]:.3f}, "
                            f"mu_sep -> {self.loss_fn.mu_sep:.4f}, lr -> {new_lr:.6f}, "
                            f"td_cooldown={level_recovery_td_cooldown}"
                        )

            # Soft landing: mode prÃ©ventif (epoch) ou rÃ©actif (divergence), une seule fois.
            soft_landing_by_epoch = (
                self.soft_landing_config.epoch is not None and (epoch + 1) >= self.soft_landing_config.epoch
            )
            soft_landing_by_div = (
                self.soft_landing_config.trigger_divergence is not None
                and level_div >= self.soft_landing_config.trigger_divergence
            )
            if not soft_landing_applied and (soft_landing_by_epoch or soft_landing_by_div):
                new_lr = max(
                    self.stability_config.min_lr,
                    self.optimizer.get_lr() * self.soft_landing_config.lr_factor,
                )
                self.optimizer.set_lr(new_lr)
                if self.soft_landing_config.freeze_mu:
                    self._freeze_mu_positions()
                soft_landing_applied = True
                if self.verbose:
                    reason = (
                        f"epoch>={self.soft_landing_config.epoch}"
                        if soft_landing_by_epoch
                        else f"level_div>={self.soft_landing_config.trigger_divergence:.3f}"
                    )
                    print(
                        f"[soft-landing] reason={reason}, epoch={epoch+1}, "
                        f"lr -> {new_lr:.6f}, freeze_mu={self.soft_landing_config.freeze_mu}"
                    )

            stability_phase_ok = (not self.stability_config.phase3_only) or (phase == 3)
            cooldown_ok = (epoch - last_lr_decay_epoch) >= self.stability_config.lr_decay_cooldown_epochs
            decay_budget_ok = lr_decay_events < self.stability_config.max_lr_decay_events
            if (
                level_div > self.stability_config.level_divergence_threshold
                and stability_phase_ok
                and (
                    (not self.stability_config.skip_lr_decay_if_recovery_triggered)
                    or (not recovery_triggered_this_epoch)
                )
            ):
                divergence_streak += 1
                if cooldown_ok and decay_budget_ok:
                    lr_before_stability = float(self.optimizer.get_lr())
                    new_lr = max(
                        self.stability_config.min_lr,
                        lr_before_stability * self.stability_config.lr_decay_on_divergence,
                    )
                    self.optimizer.set_lr(new_lr)
                    lr_decay_events += 1
                    last_lr_decay_epoch = epoch
                    self.history['events'].append(
                        {
                            'type': 'stability_lr_decay',
                            'epoch': int(epoch + 1),
                            'level_divergence': float(level_div),
                            'lr_before': lr_before_stability,
                            'lr_after': float(new_lr),
                            'event_index': int(lr_decay_events),
                        }
                    )
                    if self.verbose:
                        print(
                            f"[stability] level_div={level_div:.3f} > "
                            f"{self.stability_config.level_divergence_threshold:.3f}; "
                            f"lr -> {new_lr:.6f} (event {lr_decay_events}/{self.stability_config.max_lr_decay_events})"
                        )
            else:
                divergence_streak = 0

            # Affichage
            if self.verbose:
                msg = f"Epoch {epoch+1}: loss={epoch_loss:.3f}, acc={accuracy:.3f}"
                msg += f", level_acc={['{:.3f}'.format(a) for a in level_accuracy]}"
                msg += f", level_div={level_div:.3f}, lr={self.optimizer.get_lr():.6f}"
                msg += f", max_steps={self.model.integrator.max_steps}"
                if mean_convergence_fraction is not None:
                    msg += f", conv_frac={mean_convergence_fraction:.3f}"
                if val_accuracy is not None:
                    msg += f", val_acc={val_accuracy:.3f}"
                print(msg)

            if self.stability_config.early_stop_on_divergence and divergence_streak >= self.stability_config.divergence_patience:
                if self.verbose:
                    print(
                        f"[early-stop] divergence streak={divergence_streak} "
                        f"(threshold={self.stability_config.level_divergence_threshold:.3f})"
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
        """Phase 1 : Niveaux indÃ©pendants (Î±_bu = Î±_td = 0)."""
        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(1e-6)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(1e-6)))

    def _set_coupling_phase2(self, progress):
        """
        Phase 2 : Couplage progressif (Î± croÃ®t linÃ©airement).

        Args:
            progress (float): Progression 0->1
        """
        alpha_bu = progress * self.alpha_bu_target
        alpha_td = (progress ** self.phase_config.td_warmup_power) * self.alpha_td_target

        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(alpha_bu + 1e-6)))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(alpha_td + 1e-6)))

    def _set_coupling_phase3(self, td_scale=1.0):
        """Phase 3 : Couplage complet (Î± = valeur cible)."""
        with torch.no_grad():
            self.model.coupling.log_alpha_bu.fill_(torch.log(torch.tensor(self.alpha_bu_target)))
            td_value = max(1e-6, float(self.alpha_td_target * td_scale))
            self.model.coupling.log_alpha_td.fill_(torch.log(torch.tensor(td_value)))

    def _set_phase_max_steps(self, phase):
        """Apply optional per-phase max_steps schedule."""
        configured = None
        if phase == 1:
            configured = self.phase_config.phase1_max_steps
        elif phase == 2:
            configured = self.phase_config.phase2_max_steps
        else:
            configured = self.phase_config.phase3_max_steps
        if configured is not None:
            self.model.integrator.max_steps = int(configured)

    def _freeze_mu_positions(self):
        """GÃ¨le les positions des attracteurs (mu) pour limiter la dÃ©rive tardive."""
        for level in self.model.levels:
            if hasattr(level, "freeze_positions"):
                level.freeze_positions()
                continue
            for c in range(level.n_classes):
                for attractor in level.attractors[str(c)]:
                    attractor.position.requires_grad_(False)

    def _detect_stuck_level(self, level_accuracy, level_recovery_streak):
        """Detecte un niveau bloque proche du hasard en phase 3."""
        stuck_level_idx = None
        lowest_acc = 1.0
        target_idx = self.level_recovery_config.target_level_idx
        for idx, (level, acc) in enumerate(zip(self.model.levels, level_accuracy)):
            if target_idx is not None and idx != target_idx:
                level_recovery_streak[idx] = 0
                continue
            chance = 1.0 / float(level.n_classes)
            if abs(acc - chance) <= self.level_recovery_config.chance_tolerance:
                level_recovery_streak[idx] += 1
            else:
                level_recovery_streak[idx] = 0
            if level_recovery_streak[idx] >= self.level_recovery_config.patience and acc < lowest_acc:
                stuck_level_idx = idx
                lowest_acc = acc
        return stuck_level_idx

    def _other_levels_are_strong(self, level_accuracy, stuck_level_idx):
        """Check differential gate: other levels learn while one level is stuck."""
        other_acc = [acc for idx, acc in enumerate(level_accuracy) if idx != stuck_level_idx]
        if not other_acc:
            return False
        mean_other_acc = sum(other_acc) / len(other_acc)
        return mean_other_acc >= self.level_recovery_config.other_levels_strong_threshold

    def _has_time_for_recovery(self, epoch):
        """Avoid late triggers when too few epochs remain to have useful effect."""
        epochs_remaining = int(self.phase_config.n_epochs) - int(epoch + 1)
        return epochs_remaining >= int(self.level_recovery_config.min_epochs_remaining_to_trigger)

    def _apply_level_recovery(self, level_idx, X_train_t, y_train_t):
        """Re-anchor localement un niveau bloque sur des prototypes supervises."""
        level = self.model.levels[level_idx]
        with torch.no_grad():
            level_states = self.model.spaces(X_train_t)[level_idx]
            for class_idx in range(level.n_classes):
                class_attractors = level.attractors[str(class_idx)]
                if len(class_attractors) == 0:
                    continue
                class_mask = (y_train_t == class_idx)
                if torch.sum(class_mask).item() == 0:
                    continue
                class_points = level_states[class_mask]
                class_center = torch.mean(class_points, dim=0)
                class_std = torch.std(class_points, dim=0, unbiased=False)
                radial_scale = max(float(torch.mean(class_std).item()), self.level_recovery_config.jitter_std)
                for attractor in class_attractors:
                    jitter = radial_scale * torch.randn_like(class_center)
                    target = class_center + jitter
                    old_pos = attractor.position.detach()
                    attractor.position.copy_(0.7 * old_pos + 0.3 * target)

    def _compute_level_attractor_diagnostics(self):
        """Compute per-level collapse/separation metrics from current attractors."""
        diagnostics = []
        for level_idx, level in enumerate(self.model.levels):
            class_centroids = []
            intra_spreads = []
            for class_idx in range(level.n_classes):
                class_attractors = level.attractors[str(class_idx)]
                if len(class_attractors) == 0:
                    continue
                positions = torch.stack([a.position.detach() for a in class_attractors], dim=0)
                centroid = torch.mean(positions, dim=0)
                spread = torch.sqrt(torch.mean(torch.sum((positions - centroid) ** 2, dim=1)))
                class_centroids.append(centroid)
                intra_spreads.append(spread)

            if len(class_centroids) >= 2:
                centroids = torch.stack(class_centroids, dim=0)
                centroid_dists = torch.cdist(centroids, centroids, p=2)
                eye = torch.eye(centroid_dists.shape[0], device=centroid_dists.device, dtype=torch.bool)
                inter_vals = centroid_dists[~eye]
                inter_min = float(torch.min(inter_vals).item())
                inter_mean = float(torch.mean(inter_vals).item())
            else:
                inter_min = 0.0
                inter_mean = 0.0

            if len(intra_spreads) > 0:
                intra = torch.stack(intra_spreads)
                intra_mean = float(torch.mean(intra).item())
                intra_max = float(torch.max(intra).item())
                intra_min = float(torch.min(intra).item())
            else:
                intra_mean = 0.0
                intra_max = 0.0
                intra_min = 0.0

            separation_ratio = inter_min / (intra_mean + 1e-8) if intra_mean > 0 else 0.0
            diagnostics.append({
                'level_idx': int(level_idx),
                'intra_spread_mean': float(intra_mean),
                'intra_spread_min': float(intra_min),
                'intra_spread_max': float(intra_max),
                'inter_centroid_dist_min': float(inter_min),
                'inter_centroid_dist_mean': float(inter_mean),
                'separation_ratio': float(separation_ratio),
            })
        return diagnostics

    def plot_history(self):
        """Visualise l'historique d'entraÃ®nement."""
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
        axes[1, 1].axvspan(0, self.phase_config.phase1_epochs, alpha=0.2, color='red', label='Phase 1')
        axes[1, 1].axvspan(self.phase_config.phase1_epochs, self.phase_config.phase1_epochs + self.phase_config.phase2_epochs,
                          alpha=0.2, color='orange', label='Phase 2')
        axes[1, 1].axvspan(self.phase_config.phase1_epochs + self.phase_config.phase2_epochs, self.phase_config.n_epochs,
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

