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


class ODEIntegrator(nn.Module):
    """
    Intégrateur ODE pour le système dynamique hiérarchique.

    Équation générale :
        dx/dt = F(x, t) - γ * dx/dt

    Avec amortissement, en régime sur-amorti :
        dx/dt = (1/γ) * F(x, t)
    """

    def __init__(self, levels, coupling, gamma=1.0, method='euler', dt=0.1, max_steps=100, tol=1e-4):
        """
        Args:
            levels (list[Level]): Niveaux hiérarchiques
            coupling (BidirectionalCoupling): Module de couplage
            gamma (float): Coefficient d'amortissement
            method (str): 'euler', 'rk4', ou 'adjoint'
            dt (float): Pas de temps
            max_steps (int): Nombre maximum d'itérations
            tol (float): Tolérance de convergence ||dx/dt|| < tol
        """
        super().__init__()

        self.levels = nn.ModuleList(levels)
        self.coupling = coupling
        self.gamma = gamma
        self.method = method
        self.dt = dt
        self.max_steps = max_steps
        self.tol = tol

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

            # Critère de convergence : ||dx/dt|| < tol à tous les niveaux
            max_velocity = 0.0
            for x_old, x_new in zip(states, new_states):
                dx = x_new - x_old
                velocity = torch.norm(dx) / self.dt
                max_velocity = max(max_velocity, velocity.item())

            if max_velocity < self.tol:
                converged = True
                states = new_states
                if return_trajectory:
                    trajectory.append(states)
                break

            states = new_states

            if return_trajectory:
                trajectory.append(states)

        return states, step + 1, converged, trajectory

    def integrate_adjoint(self, initial_states):
        """
        Intégration avec méthode adjointe (pour backprop efficace en mémoire).

        Nécessite torchdiffeq. Wrapper pour compatibilité future.

        Args:
            initial_states (list[torch.Tensor]): États initiaux

        Returns:
            tuple: (final_states, n_steps, converged, trajectory)
        """
        # TODO: Implémenter avec torchdiffeq.odeint_adjoint
        # Pour l'instant, fallback sur RK4
        print("Warning: Adjoint method not implemented, using RK4")
        return self.integrate(initial_states, return_trajectory=False)

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
