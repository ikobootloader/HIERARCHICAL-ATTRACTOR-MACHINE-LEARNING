"""
Module ODEIntegrator.

Implémente l'intégration de l'équation du mouvement :
    dx^(l)/dt = F_intra + F_bu + F_td - γ * dx^(l)/dt

Méthodes :
- Euler explicite (rapide, pour prototypage)
- RK4 (précis, recommandé)
- Interface pour méthode adjointe (torchdiffeq, mémoire efficace)
"""

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint_adjoint as _odeint_adjoint
except Exception:  # pragma: no cover - optional dependency
    _odeint_adjoint = None


class ODEIntegrator(nn.Module):
    """
    Intégrateur ODE pour le système dynamique hiérarchique.

    Équation générale :
        dx/dt = F(x, t) - γ * dx/dt

    Avec amortissement, en régime sur-amorti :
        dx/dt = (1/γ) * F(x, t)
    """

    def __init__(
        self,
        levels,
        coupling,
        gamma=1.0,
        method='euler',
        dt=0.1,
        max_steps=100,
        tol=1e-4,
        convergence_check_every=5,
    ):
        """
        Args:
            levels (list[Level]): Niveaux hiérarchiques
            coupling (BidirectionalCoupling): Module de couplage
            gamma (float): Coefficient d'amortissement
            method (str): 'euler', 'rk4', ou 'adjoint'
            dt (float): Pas de temps
            max_steps (int): Nombre maximum d'itérations
            tol (float|None): Tolérance de convergence ||dx/dt|| < tol (None = désactivé)
            convergence_check_every (int): Fréquence des checks de convergence (en pas)
        """
        super().__init__()

        self.levels = nn.ModuleList(levels)
        self.coupling = coupling
        self.gamma = gamma
        self.method = method
        self.dt = dt
        self.max_steps = max_steps
        self.tol = tol
        self.convergence_check_every = max(1, int(convergence_check_every))
        self._warned_missing_adjoint = False

    def compute_forces(self, states):
        """
        Calcule toutes les forces à l'instant t.

        Args:
            states (list[torch.Tensor]): États [x^(0), ..., x^(L)]

        Returns:
            list[torch.Tensor]: Forces [F^(0), ..., F^(L)]
        """
        forces = []

        for l, level in enumerate(self.levels):
            # Force intra-niveau
            f_intra = level.intra_level_force(states[l])

            # Forces de couplage
            f_coupling = self.coupling.total_force(states, l)

            # Force totale
            f_total = f_intra + f_coupling

            forces.append(f_total)

        return forces

    def velocity_field(self, states):
        """
        Calcule dx/dt = (1/γ) * F(x).

        Args:
            states (list[torch.Tensor]): États courants

        Returns:
            list[torch.Tensor]: Vitesses [dx^(0)/dt, ..., dx^(L)/dt]
        """
        forces = self.compute_forces(states)
        velocities = [f / self.gamma for f in forces]
        return velocities

    def step_euler(self, states):
        """
        Un pas d'intégration Euler : x(t+dt) = x(t) + dt * dx/dt.

        Args:
            states (list[torch.Tensor]): États courants

        Returns:
            list[torch.Tensor]: Nouveaux états
        """
        velocities = self.velocity_field(states)

        new_states = []
        for x, v in zip(states, velocities):
            x_new = x + self.dt * v
            new_states.append(x_new)

        return new_states

    def step_rk4(self, states):
        """
        Un pas d'intégration Runge-Kutta 4.

        Args:
            states (list[torch.Tensor]): États courants

        Returns:
            list[torch.Tensor]: Nouveaux états
        """
        # k1 = F(x)
        k1 = self.velocity_field(states)

        # k2 = F(x + dt/2 * k1)
        states_k2 = [x + 0.5 * self.dt * v for x, v in zip(states, k1)]
        k2 = self.velocity_field(states_k2)

        # k3 = F(x + dt/2 * k2)
        states_k3 = [x + 0.5 * self.dt * v for x, v in zip(states, k2)]
        k3 = self.velocity_field(states_k3)

        # k4 = F(x + dt * k3)
        states_k4 = [x + self.dt * v for x, v in zip(states, k3)]
        k4 = self.velocity_field(states_k4)

        # x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        new_states = []
        for x, v1, v2, v3, v4 in zip(states, k1, k2, k3, k4):
            x_new = x + (self.dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
            new_states.append(x_new)

        return new_states

    def integrate(self, initial_states, return_trajectory=False):
        """
        Intègre le système jusqu'à convergence.

        Args:
            initial_states (list[torch.Tensor]): États initiaux
            return_trajectory (bool): Si True, retourne toute la trajectoire

        Returns:
            tuple: (final_states, n_steps, converged, trajectory)
                - final_states : États finaux
                - n_steps : Nombre d'itérations
                - converged : True si convergence atteinte
                - trajectory : Liste des états si return_trajectory=True
        """
        if self.method == 'adjoint':
            return self.integrate_adjoint(initial_states, return_trajectory=return_trajectory)

        states = initial_states
        trajectory = [states] if return_trajectory else []

        converged = False

        for step in range(self.max_steps):
            # Un pas d'intégration
            if self.method == 'euler':
                new_states = self.step_euler(states)
            elif self.method == 'rk4':
                new_states = self.step_rk4(states)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            if self.tol is not None and ((step + 1) % self.convergence_check_every == 0):
                # Reduce on device and sync to host only every N steps.
                max_velocity = None
                for x_old, x_new in zip(states, new_states):
                    dx = x_new - x_old
                    velocity = torch.norm(dx) / self.dt
                    max_velocity = velocity if max_velocity is None else torch.maximum(max_velocity, velocity)

                if float(max_velocity.item()) < float(self.tol):
                    converged = True
                    states = new_states
                    if return_trajectory:
                        trajectory.append(states)
                    break

            states = new_states

            if return_trajectory:
                trajectory.append(states)

        return states, step + 1, converged, trajectory

    def _flatten_states(self, states):
        """Concatène [x^(0),...,x^(L)] en un tenseur 2D (batch, total_dim)."""
        return torch.cat(states, dim=1)

    def _unflatten_states(self, flat, state_dims):
        """Découpe un tenseur 2D (batch, total_dim) en liste d'états par niveau."""
        parts = []
        offset = 0
        for dim in state_dims:
            parts.append(flat[:, offset:offset + dim])
            offset += dim
        return parts

    def _compute_convergence_step(self, flat_traj):
        """
        Détecte le premier pas où ||dx/dt|| < tol.

        Args:
            flat_traj (torch.Tensor): (n_times, batch, total_dim)

        Returns:
            tuple: (step_idx, converged)
        """
        n_times = flat_traj.shape[0]
        if self.tol is None:
            return n_times - 1, False
        for i in range(1, n_times):
            dx = flat_traj[i] - flat_traj[i - 1]
            velocity = torch.norm(dx) / self.dt
            if velocity.item() < self.tol:
                return i, True
        return n_times - 1, False

    def integrate_adjoint(self, initial_states, return_trajectory=False):
        """
        Intégration avec méthode adjointe (pour backprop efficace en mémoire).

        Nécessite torchdiffeq. Wrapper pour compatibilité future.

        Args:
            initial_states (list[torch.Tensor]): États initiaux
            return_trajectory (bool): Si True, retourne la trajectoire discrète

        Returns:
            tuple: (final_states, n_steps, converged, trajectory)
        """
        if _odeint_adjoint is None:
            if not self._warned_missing_adjoint:
                print("Warning: torchdiffeq not available, fallback to RK4")
                self._warned_missing_adjoint = True
            prev_method = self.method
            self.method = 'rk4'
            try:
                return self.integrate(initial_states, return_trajectory=return_trajectory)
            finally:
                self.method = prev_method

        state_dims = [x.shape[1] for x in initial_states]
        y0 = self._flatten_states(initial_states)
        t = torch.linspace(
            0.0,
            self.dt * self.max_steps,
            self.max_steps + 1,
            device=y0.device,
            dtype=y0.dtype
        )

        def rhs(_t, y_flat):
            states = self._unflatten_states(y_flat, state_dims)
            velocities = self.velocity_field(states)
            return self._flatten_states(velocities)

        # Adjoint method for memory-efficient backprop through ODE solve
        flat_traj = _odeint_adjoint(
            rhs,
            y0,
            t,
            method='rk4',
            options={'step_size': self.dt},
            adjoint_params=tuple(self.parameters())
        )  # (n_times, batch, total_dim)

        step_idx, converged = self._compute_convergence_step(flat_traj)
        final_flat = flat_traj[step_idx]
        final_states = self._unflatten_states(final_flat, state_dims)

        trajectory = []
        if return_trajectory:
            for i in range(step_idx + 1):
                trajectory.append(self._unflatten_states(flat_traj[i], state_dims))

        return final_states, step_idx, converged, trajectory

    def compute_dynamic_loss(self, trajectory):
        """
        Calcule la loss de dynamique L_dyn = (1/T) ∫ Σ_l ||dx^(l)/dt||² dt.

        Encourage la convergence rapide.

        Args:
            trajectory (list): Liste des états à chaque pas de temps

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

            for x_old, x_new in zip(states_old, states_new):
                dx = x_new - x_old
                loss += torch.sum(dx ** 2)

        loss = loss / (n_steps * self.dt)
        return loss

    def set_damping(self, gamma):
        """Modifie le coefficient d'amortissement."""
        self.gamma = gamma

    def set_tolerance(self, tol):
        """Modifie la tolérance de convergence."""
        self.tol = tol
