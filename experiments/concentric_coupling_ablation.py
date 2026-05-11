"""
Ablation couplage sur dataset concentrique (anneaux alternÃ©s).

Objectif: tester un cas oÃ¹ le contexte global doit aider la dÃ©cision locale.
Compare:
- Independent: alpha_bu=0.0, alpha_td=0.0
- Coupled: alpha_bu=1.0, alpha_td=1.0
"""

import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from haml import HAML


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_concentric_alternating(
    n_samples=3000,
    radii=(1.0, 2.0, 3.0, 4.0),
    noise_r=0.12,
    noise_xy=0.05,
    seed=42,
):
    """
    GÃ©nÃ¨re des anneaux concentriques avec classes alternÃ©es:
    classe 0 sur anneaux 0,2 ; classe 1 sur anneaux 1,3.
    """
    rng = np.random.RandomState(seed)
    n_rings = len(radii)
    per_ring = n_samples // n_rings
    X_parts = []
    y_parts = []

    for i, r in enumerate(radii):
        theta = rng.uniform(0, 2 * np.pi, size=per_ring)
        rr = r + rng.normal(0.0, noise_r, size=per_ring)
        x = rr * np.cos(theta) + rng.normal(0.0, noise_xy, size=per_ring)
        y = rr * np.sin(theta) + rng.normal(0.0, noise_xy, size=per_ring)
        X_parts.append(np.column_stack([x, y]))
        y_parts.append(np.full(per_ring, i % 2, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def train_and_eval(
    name,
    alpha_bu,
    alpha_td,
    X_train,
    y_train,
    X_test,
    y_test,
    seed=42,
    n_attractors_per_class=3,
    n_epochs=12,
    max_steps=30,
    phase1_epochs=None,
    phase2_epochs=None,
):
    set_seed(seed)
    model = HAML(
        n_levels=3,
        n_attractors_per_class=n_attractors_per_class,
        alpha_bu=alpha_bu,
        alpha_td=alpha_td,
        max_steps=max_steps,
        lr=0.01,
        n_epochs=n_epochs,
        batch_size=128,
        train_on_fit=False,
        device="cpu",
    )

    model.fit(X_train, y_train)
    start = time.time()
    if phase1_epochs is None or phase2_epochs is None:
        history = model.train_model(X_train, y_train)
    else:
        from haml.training import HAMLLoss, ConstrainedOptimizer, HAMLTrainer

        optimizer = ConstrainedOptimizer(model.parameters(), lr=model.lr)
        trainer = HAMLTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=HAMLLoss(mu_sep=model.mu_sep, mu_dyn=model.mu_dyn),
            n_epochs=n_epochs,
            batch_size=model.batch_size,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            td_warmup_power=2.0,
            level_divergence_threshold=0.15,
            divergence_patience=2,
            lr_decay_on_divergence=0.5,
            min_lr=1e-4,
            early_stop_on_divergence=False,
            adaptive_mu_sep=False,
            adaptive_mu_sep_phase3_only=True,
            mu_sep_trigger_divergence=0.11,
            mu_sep_patience=2,
            mu_sep_growth_factor=1.05,
            mu_sep_max=1.0,
            soft_landing_epoch=None,
            soft_landing_trigger_divergence=None,
            soft_landing_lr_factor=0.2,
            soft_landing_freeze_mu=True,
            collapse_guard_enabled=True,
            collapse_guard_start_epoch=18,
            collapse_guard_delta_div_threshold=0.10,
            collapse_guard_mu_sep_boost=1.5,
            collapse_guard_lr_factor=0.3,
            device=model.device,
            verbose=True,
        )
        history = trainer.train(X_train, y_train)
    train_time = time.time() - start
    test_acc = model.score(X_test, y_test)

    return model, {
        "name": name,
        "alpha_bu": alpha_bu,
        "alpha_td": alpha_td,
        "n_attractors_per_class": n_attractors_per_class,
        "n_epochs": n_epochs,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "train_accuracy_by_epoch": history["accuracy"],
        "test_accuracy": float(test_acc),
        "train_time_sec": float(train_time),
    }


def plot_decision(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.28, levels=[-0.5, 0.5, 1.5], cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="coolwarm", alpha=0.65, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


def main():
    seed = 42
    set_seed(seed)

    X, y = make_concentric_alternating(
        n_samples=3200,
        radii=(1.0, 2.0, 3.0, 4.0),
        noise_r=0.12,
        noise_xy=0.05,
        seed=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Baseline independent
    model_ind, res_ind = train_and_eval(
        "independent",
        0.0,
        0.0,
        X_train,
        y_train,
        X_test,
        y_test,
        seed=seed,
        n_attractors_per_class=3,
        n_epochs=12,
    )

    # Coupled tuned + defensive stability controls
    model_cpl_tuned, res_cpl_tuned = train_and_eval(
        "coupled_tuned_stable",
        0.5,
        1.5,
        X_train,
        y_train,
        X_test,
        y_test,
        seed=seed,
        n_attractors_per_class=5,
        n_epochs=25,
        max_steps=40,
        phase1_epochs=5,
        phase2_epochs=8,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision(
        axes[0],
        model_ind,
        X_test,
        y_test,
        f"Independent (test={res_ind['test_accuracy']:.3f})",
    )
    plot_decision(
        axes[1],
        model_cpl_tuned,
        X_test,
        y_test,
        f"Coupled tuned stable (test={res_cpl_tuned['test_accuracy']:.3f})",
    )
    plt.tight_layout()
    out_png = "concentric_coupling_ablation.png"
    fig.savefig(out_png, dpi=140)

    summary = {
        "dataset": {
            "name": "concentric_alternating",
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "seed": seed,
        },
        "independent": res_ind,
        "coupled_tuned": res_cpl_tuned,
        "delta_test_accuracy_points_tuned_vs_independent":
            (res_cpl_tuned["test_accuracy"] - res_ind["test_accuracy"]) * 100.0,
        "delta_train_time_sec_tuned_vs_independent":
            res_cpl_tuned["train_time_sec"] - res_ind["train_time_sec"],
        "figure": out_png,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

