"""Short MNIST probe for validating coupled recovery behavior before full campaign."""

import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def load_mnist_subset(n_train, n_test, seed):
    cache_dir = os.path.join(os.getcwd(), ".sklearn_data")
    os.makedirs(cache_dir, exist_ok=True)

    mnist = fetch_openml("mnist_784", version=1, parser="auto", data_home=cache_dir)
    X = mnist.data.to_numpy()
    y = mnist.target.to_numpy().astype(int)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n_train + n_test, replace=False)
    X = X[idx]
    y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def build_trainer(model, n_epochs, phase1_epochs, phase2_epochs):
    return HAMLTrainer(
        model=model,
        optimizer=ConstrainedOptimizer(model.parameters(), lr=model.lr),
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
            phase3_only=True,
            lr_decay_cooldown_epochs=2,
            max_lr_decay_events=4,
            skip_lr_decay_if_recovery_triggered=True,
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
            start_epoch=8,
            delta_div_threshold=0.10,
            mu_sep_boost=1.5,
            lr_factor=0.3,
        ),
        level_recovery_config=LevelRecoveryConfig(
            enabled=True,
            chance_tolerance=0.12,  # chance(10 classes)=0.10, threshold=0.22
            patience=3,
            max_triggers=1,
            require_divergence=0.11,
            other_levels_strong_threshold=0.20,
            min_epochs_remaining_to_trigger=2,
            mu_sep_boost=1.2,
            lr_factor=0.8,
            jitter_std=0.01,
            target_level_idx=None,
        ),
        device=model.device,
        verbose=True,
    )


def phase_accuracy(history, phase1_epochs, phase2_epochs):
    idx_p1 = phase1_epochs - 1
    idx_p2 = phase1_epochs + phase2_epochs - 1
    idx_p3 = len(history["accuracy"]) - 1
    return {
        "phase1_end_train_accuracy": float(history["accuracy"][idx_p1]),
        "phase2_end_train_accuracy": float(history["accuracy"][idx_p2]),
        "phase3_end_train_accuracy": float(history["accuracy"][idx_p3]),
    }


def main():
    parser = argparse.ArgumentParser(description="MNIST short probe with recovery diagnostics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--phase1-epochs", type=int, default=2)
    parser.add_argument("--phase2-epochs", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--json-out",
        type=str,
        default="github_app/experiments/diag_mnist_seed42_n5000_e10_mutex.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    X_train, X_test, y_train, y_test = load_mnist_subset(args.n_train, args.n_test, args.seed)

    model = HAML(
        n_levels=3,
        n_attractors_per_class=2,
        alpha_bu=0.5,
        alpha_td=1.5,
        max_steps=40,
        lr=0.01,
        n_epochs=args.n_epochs,
        batch_size=128,
        train_on_fit=False,
        device=args.device,
    )
    model.fit(X_train, y_train)
    trainer = build_trainer(model, args.n_epochs, args.phase1_epochs, args.phase2_epochs)

    start = time.time()
    history = trainer.train(X_train, y_train)
    train_time_sec = time.time() - start
    test_acc = model.score(X_test, y_test)

    phase_metrics = phase_accuracy(history, args.phase1_epochs, args.phase2_epochs)
    phase3_start = args.phase1_epochs + args.phase2_epochs
    level_accuracy_phase3 = [
        {
            "epoch": int(epoch_idx + 1),
            "level_accuracy": [float(v) for v in row],
        }
        for epoch_idx, row in enumerate(history["level_accuracy"])
        if (epoch_idx + 1) > phase3_start
    ]

    summary = {
        "dataset": "mnist_784",
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "n_epochs": int(args.n_epochs),
        "phase1_epochs": int(args.phase1_epochs),
        "phase2_epochs": int(args.phase2_epochs),
        "phase3_epochs": int(args.n_epochs - args.phase1_epochs - args.phase2_epochs),
        "test_accuracy": float(test_acc),
        "train_time_sec": float(train_time_sec),
        "phase_accuracy": phase_metrics,
        "level_accuracy_phase3": level_accuracy_phase3,
        "events": history.get("events", []),
        "recovery_events": [e for e in history.get("events", []) if e.get("type") == "level_recovery"],
        "stability_decay_events": [e for e in history.get("events", []) if e.get("type") == "stability_lr_decay"],
        "level_recovery_config": {
            "chance_tolerance": 0.12,
            "stuck_threshold_from_chance": 0.10 + 0.12,
            "other_levels_strong_threshold": 0.20,
        },
    }

    out_path = pathlib.Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
