import pathlib
import sys

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Priorise le package haml de github_app (version active du projet).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml import HAML
from haml.training import ConstrainedOptimizer, HAMLLoss, HAMLTrainer, PhaseConfig


def test_phase_integrator_method_schedule_switches_methods():
    X, y = make_moons(n_samples=240, noise=0.2, random_state=0)
    X = StandardScaler().fit_transform(X)

    model = HAML(
        n_levels=3,
        level_dims=[2, 2],
        n_attractors_per_class=2,
        alpha_bu=0.5,
        alpha_td=1.5,
        use_vectorized_levels=True,
        max_steps=6,
        n_epochs=2,
        batch_size=64,
        train_on_fit=False,
        device="cpu",
    )
    model.fit(X, y)

    original_euler = model.integrator.step_euler
    original_rk4 = model.integrator.step_rk4
    calls = {"euler": 0, "rk4": 0}

    def wrapped_euler(states):
        calls["euler"] += 1
        return original_euler(states)

    def wrapped_rk4(states):
        calls["rk4"] += 1
        return original_rk4(states)

    model.integrator.step_euler = wrapped_euler
    model.integrator.step_rk4 = wrapped_rk4

    trainer = HAMLTrainer(
        model=model,
        optimizer=ConstrainedOptimizer(model.parameters(), lr=model.lr),
        loss_fn=HAMLLoss(mu_sep=model.mu_sep, mu_dyn=0.0),
        phase_config=PhaseConfig(
            n_epochs=2,
            batch_size=64,
            phase1_epochs=1,
            phase2_epochs=1,
            phase1_integrator_method="euler",
            phase2_integrator_method="rk4",
        ),
        device="cpu",
        verbose=False,
    )
    trainer.train(X, y)

    assert calls["euler"] > 0
    assert calls["rk4"] > 0
