"""
Seed-wise diagnostics for concentric coupling instability analysis.

Exports per-seed traces and summaries:
- level_divergence
- mu_sep
- learning rate
- level_accuracy
- attractor state statistics
"""

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Priorise le package local github_app/haml.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml import HAML
from haml.training import (
    AdaptiveMuSepConfig,
    CollapseGuardConfig,
    ConstrainedOptimizer,
    HAMLLoss,
    HAMLTrainer,
    LevelRecoveryConfig,
    PhaseConfig,
    SoftLandingConfig,
    StabilityConfig,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_concentric_alternating(
    n_samples=3200,
    radii=(1.0, 2.0, 3.0, 4.0),
    noise_r=0.12,
    noise_xy=0.05,
    seed=42,
):
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


def make_noisy_moons_dataset(
    n_samples=3200,
    noise=0.30,
    seed=42,
):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y


def make_classification_dataset(
    n_samples=3200,
    n_classes=4,
    n_features=12,
    n_informative=6,
    n_redundant=4,
    class_sep=1.0,
    seed=42,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=None,
        class_sep=class_sep,
        random_state=seed,
    )
    return X, y


def _extract_attractor_stats(model):
    levels_stats = []
    for level_idx, level in enumerate(model.levels):
        level_data = {
            "level_idx": int(level_idx),
            "n_classes": int(level.n_classes),
            "n_attractors_per_class": int(level.n_attractors_per_class),
            "attractors": [],
        }
        for class_idx in range(level.n_classes):
            for attractor_idx, attractor in enumerate(level.attractors[str(class_idx)]):
                position = attractor.position.detach().cpu().numpy()
                level_data["attractors"].append(
                    {
                        "class_idx": int(class_idx),
                        "attractor_idx": int(attractor_idx),
                        "sigma": float(attractor.sigma.item()),
                        "rho": float(attractor.rho.item()),
                        "rho_sigma_ratio": float((attractor.rho / attractor.sigma).item()),
                        "weight": float(attractor.weight.item()),
                        "position_norm": float(np.linalg.norm(position)),
                        "position": position.tolist(),
                    }
                )
        levels_stats.append(level_data)
    return levels_stats


def _history_summary(history):
    level_div = history["level_divergence"]
    lr = history["lr"]
    mu_sep = history["mu_sep"]
    level_acc = history["level_accuracy"]

    max_delta_div = 0.0
    if len(level_div) > 1:
        max_delta_div = float(np.max(np.diff(np.array(level_div))))

    return {
        "epochs_ran": int(len(history["loss"])),
        "train_accuracy_final": float(history["accuracy"][-1]),
        "train_accuracy_max": float(np.max(history["accuracy"])),
        "level_divergence_max": float(np.max(level_div)),
        "level_divergence_final": float(level_div[-1]),
        "level_divergence_max_delta": float(max_delta_div),
        "lr_initial": float(lr[0]),
        "lr_final": float(lr[-1]),
        "mu_sep_initial": float(mu_sep[0]),
        "mu_sep_final": float(mu_sep[-1]),
        "level_accuracy_final": [float(v) for v in level_acc[-1]],
        "level_accuracy_min_overall": float(np.min(np.array(level_acc))),
        "attractor_separation_ratio_final": [
            float(level_diag["separation_ratio"])
            for level_diag in history["level_attractor_diagnostics"][-1]
        ],
        "attractor_separation_ratio_min_overall": float(
            np.min(
                [
                    level_diag["separation_ratio"]
                    for epoch_diag in history["level_attractor_diagnostics"]
                    for level_diag in epoch_diag
                ]
            )
        ),
    }


def train_configured_model(X_train, y_train, mode, seed):
    if mode == "independent":
        model = HAML(
            n_levels=3,
            n_attractors_per_class=3,
            alpha_bu=0.0,
            alpha_td=0.0,
            max_steps=30,
            lr=0.01,
            n_epochs=12,
            batch_size=128,
            train_on_fit=False,
            device="cpu",
        )
        model.fit(X_train, y_train)
        history = model.train_model(X_train, y_train)
        return model, history

    if mode == "coupled_tuned":
        model = HAML(
            n_levels=3,
            n_attractors_per_class=5,
            alpha_bu=0.5,
            alpha_td=1.5,
            max_steps=40,
            lr=0.01,
            n_epochs=25,
            batch_size=128,
            train_on_fit=False,
            device="cpu",
        )
        model.fit(X_train, y_train)
        trainer = HAMLTrainer(
            model=model,
            optimizer=ConstrainedOptimizer(model.parameters(), lr=model.lr),
            loss_fn=HAMLLoss(mu_sep=model.mu_sep, mu_dyn=model.mu_dyn),
            phase_config=PhaseConfig(
                n_epochs=25,
                batch_size=model.batch_size,
                phase1_epochs=5,
                phase2_epochs=8,
                td_warmup_power=2.0,
            ),
            stability_config=StabilityConfig(
                level_divergence_threshold=0.15,
                divergence_patience=2,
                lr_decay_on_divergence=0.5,
                min_lr=1e-4,
                early_stop_on_divergence=False,
                phase3_only=True,
                lr_decay_cooldown_epochs=2,
                max_lr_decay_events=4,
            ),
            adaptive_mu_sep_config=AdaptiveMuSepConfig(
                enabled=True,
                phase3_only=True,
                trigger_divergence=0.11,
                patience=2,
                growth_factor=1.1,
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
            level_recovery_config=LevelRecoveryConfig(
                enabled=True,
                chance_tolerance=0.02,
                patience=5,
                max_triggers=1,
                require_divergence=0.11,
                other_levels_strong_threshold=0.65,
                min_epochs_remaining_to_trigger=3,
                mu_sep_boost=1.2,
                lr_factor=0.8,
                jitter_std=0.01,
                target_level_idx=None,
            ),
            device=model.device,
            verbose=True,
        )
        history = trainer.train(X_train, y_train)
        return model, history

    raise ValueError(f"Unsupported mode: {mode}")


def run_seed(seed, n_samples, mode, dataset, moons_noise):
    set_seed(seed)
    if dataset == "concentric":
        X, y = make_concentric_alternating(n_samples=n_samples, seed=seed)
    elif dataset == "noisy_moons":
        X, y = make_noisy_moons_dataset(
            n_samples=n_samples,
            noise=moons_noise,
            seed=seed,
        )
    elif dataset == "make_classification":
        X, y = make_classification_dataset(
            n_samples=n_samples,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start = time.time()
    model, history = train_configured_model(X_train, y_train, mode=mode, seed=seed)
    duration = time.time() - start
    test_acc = model.score(X_test, y_test)

    return {
        "seed": int(seed),
        "dataset": dataset,
        "mode": mode,
        "n_samples": int(n_samples),
        "test_accuracy": float(test_acc),
        "train_time_sec": float(duration),
        "history": {
            "loss": [float(v) for v in history["loss"]],
            "accuracy": [float(v) for v in history["accuracy"]],
            "level_divergence": [float(v) for v in history["level_divergence"]],
            "lr": [float(v) for v in history["lr"]],
            "mu_sep": [float(v) for v in history["mu_sep"]],
            "level_accuracy": [[float(x) for x in row] for row in history["level_accuracy"]],
            "level_attractor_diagnostics": history["level_attractor_diagnostics"],
        },
        "history_summary": _history_summary(history),
        "attractor_stats": _extract_attractor_stats(model),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Seed-wise coupling diagnostics.")
    parser.add_argument("--mode", choices=["independent", "coupled_tuned"], default="coupled_tuned")
    parser.add_argument(
        "--dataset",
        choices=["concentric", "noisy_moons", "make_classification"],
        default="concentric",
    )
    parser.add_argument("--moons-noise", type=float, default=0.30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=3200)
    parser.add_argument(
        "--json-out",
        type=str,
        default="experiments/seedwise_coupling_diagnostics.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [args.base_seed + i for i in range(args.n_runs)]
    runs = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ({args.mode}, dataset={args.dataset}) ===")
        runs.append(
            run_seed(
                seed=seed,
                n_samples=args.n_samples,
                mode=args.mode,
                dataset=args.dataset,
                moons_noise=args.moons_noise,
            )
        )

    summary = {
        "dataset": args.dataset,
        "moons_noise": float(args.moons_noise) if args.dataset == "noisy_moons" else None,
        "mode": args.mode,
        "seeds": seeds,
        "n_samples": int(args.n_samples),
        "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in runs])),
        "test_accuracy_std": float(np.std([r["test_accuracy"] for r in runs], ddof=0)),
        "test_accuracy_min": float(np.min([r["test_accuracy"] for r in runs])),
        "test_accuracy_max": float(np.max([r["test_accuracy"] for r in runs])),
        "train_time_sec_mean": float(np.mean([r["train_time_sec"] for r in runs])),
        "train_time_sec_std": float(np.std([r["train_time_sec"] for r in runs], ddof=0)),
    }
    payload = {"summary": summary, "runs": runs}

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
