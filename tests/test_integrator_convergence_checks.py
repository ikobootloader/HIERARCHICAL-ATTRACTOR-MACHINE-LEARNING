import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml.dynamics.integrator import ODEIntegrator


class _LinearLevel(nn.Module):
    def intra_level_force(self, x):
        # Stable linear dynamics toward zero.
        return -x


class _ZeroCoupling(nn.Module):
    def total_force(self, states, _level_idx):
        return torch.zeros_like(states[0])


def _legacy_integrate(integrator, initial_states, return_trajectory=False):
    states = initial_states
    trajectory = [states] if return_trajectory else []
    converged = False

    for step in range(integrator.max_steps):
        if integrator.method == "euler":
            new_states = integrator.step_euler(states)
        elif integrator.method == "rk4":
            new_states = integrator.step_rk4(states)
        else:
            raise ValueError("Unknown method")

        max_velocity = 0.0
        for x_old, x_new in zip(states, new_states):
            dx = x_new - x_old
            velocity = torch.norm(dx) / integrator.dt
            max_velocity = max(max_velocity, velocity.item())

        if integrator.tol is not None and max_velocity < integrator.tol:
            converged = True
            states = new_states
            if return_trajectory:
                trajectory.append(states)
            break

        states = new_states
        if return_trajectory:
            trajectory.append(states)

    return states, step + 1, converged, trajectory


def test_k1_matches_legacy_behavior():
    level = _LinearLevel()
    coupling = _ZeroCoupling()
    integrator = ODEIntegrator(
        levels=[level],
        coupling=coupling,
        method="rk4",
        dt=0.1,
        max_steps=30,
        tol=1e-4,
        convergence_check_every=1,
    )
    initial = [torch.tensor([[3.0, -1.0]], dtype=torch.float32)]

    new_states, new_steps, new_conv, _ = integrator.integrate(initial, return_trajectory=False)
    old_states, old_steps, old_conv, _ = _legacy_integrate(integrator, initial, return_trajectory=False)

    assert new_steps == old_steps
    assert new_conv == old_conv
    assert torch.allclose(new_states[0], old_states[0], atol=1e-7, rtol=1e-7)


def test_tol_none_disables_early_stopping():
    level = _LinearLevel()
    coupling = _ZeroCoupling()
    integrator = ODEIntegrator(
        levels=[level],
        coupling=coupling,
        method="euler",
        dt=0.1,
        max_steps=7,
        tol=None,
        convergence_check_every=1,
    )
    initial = [torch.tensor([[1.0, 1.0]], dtype=torch.float32)]
    _, steps, converged, _ = integrator.integrate(initial, return_trajectory=False)
    assert steps == 7
    assert converged is False
