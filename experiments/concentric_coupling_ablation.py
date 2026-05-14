"""
Coupling ablation on a concentric alternating-rings dataset.

Objective: test a setting where global context should help local decisions.
Compares:
- Independent: alpha_bu=0.0, alpha_td=0.0
- Coupled tuned: alpha_bu=0.5, alpha_td=1.5
"""

import argparse
import json
import random
import time
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Priorise le package local github_app/haml.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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
    """Generate concentric rings with alternating class labels."""
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
        from haml.training import (
            HAMLLoss,
            ConstrainedOptimizer,
            HAMLTrainer,
            PhaseConfig,
            StabilityConfig,
            AdaptiveMuSepConfig,
            SoftLandingConfig,
            CollapseGuardConfig,
        )

        optimizer = ConstrainedOptimizer(model.parameters(), lr=model.lr)
        trainer = HAMLTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=HAMLLoss(mu_sep=model.mu_sep, mu_dyn=model.mu_dyn),
            phase_config=PhaseConfig(
                n_epochs=n_epochs,
                batch_size=model.batch_size,
                phase1_epochs=phase1_epochs,
                phase2_epochs=phase2_epochs,
                td_warmup_power=2.0,
            ),
            stability_config=StabilityConfig(
                level_divergence_threshold=0.15,
                divergence_patience=2,
                lr_decay_on_divergence=0.5,
                min_lr=1e-4,
                early_stop_on_divergence=False,
            ),
            adaptive_mu_sep_config=AdaptiveMuSepConfig(
                enabled=False,
                phase3_only=True,
                trigger_divergence=0.11,
                patience=2,
                growth_factor=1.05,
                max_value=1.0,
            ),
            soft_landing_config=SoftLandingConfig(
                epoch=None,
                trigger_divergence=None,
                lr_factor=0.2,
                freeze_mu=True,
            ),
            collapse_guard_config=CollapseGuardConfig(
                enabled=True,
                start_epoch=18,
                delta_div_threshold=0.10,
                mu_sep_boost=1.5,
                lr_factor=0.3,
            ),
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


def summarize_runs(runs):
    test_accuracies = np.array([run["test_accuracy"] for run in runs], dtype=float)
    train_times = np.array([run["train_time_sec"] for run in runs], dtype=float)
    return {
        "n_runs": int(len(runs)),
        "test_accuracy_mean": float(test_accuracies.mean()),
        "test_accuracy_std": float(test_accuracies.std(ddof=0)),
        "test_accuracy_min": float(test_accuracies.min()),
        "test_accuracy_max": float(test_accuracies.max()),
        "train_time_sec_mean": float(train_times.mean()),
        "train_time_sec_std": float(train_times.std(ddof=0)),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Concentric coupling ablation.")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of seeds to run.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for reproducibility.")
    parser.add_argument("--n-samples", type=int, default=3200, help="Number of generated samples.")
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help="Save decision boundaries for first run.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="experiments/concentric_coupling_ablation_summary.json",
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [args.base_seed + i for i in range(args.n_runs)]

    independent_runs = []
    coupled_runs = []
    figure_payload = None

    for run_index, seed in enumerate(seeds):
        set_seed(seed)
        X, y = make_concentric_alternating(
            n_samples=args.n_samples,
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

        independent_runs.append({"seed": seed, **res_ind})
        coupled_runs.append({"seed": seed, **res_cpl_tuned})

        if run_index == 0 and args.save_figure:
            figure_payload = (model_ind, model_cpl_tuned, X_test, y_test, res_ind, res_cpl_tuned)

    independent_stats = summarize_runs(independent_runs)
    coupled_stats = summarize_runs(coupled_runs)

    summary = {
        "dataset": {
            "name": "concentric_alternating",
            "n_samples": int(args.n_samples),
            "train": int(args.n_samples * 0.75),
            "test": int(args.n_samples * 0.25),
            "seeds": seeds,
        },
        "independent_runs": independent_runs,
        "coupled_tuned_runs": coupled_runs,
        "independent_stats": independent_stats,
        "coupled_tuned_stats": coupled_stats,
        "delta_test_accuracy_points_mean": (
            coupled_stats["test_accuracy_mean"] - independent_stats["test_accuracy_mean"]
        ) * 100.0,
        "delta_train_time_sec_mean": (
            coupled_stats["train_time_sec_mean"] - independent_stats["train_time_sec_mean"]
        ),
        "figure": None,
    }

    if figure_payload is not None:
        model_ind, model_cpl_tuned, X_test, y_test, res_ind, res_cpl_tuned = figure_payload
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
        summary["figure"] = out_png

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
