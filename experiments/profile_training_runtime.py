"""Profile HAML training runtime (cProfile + optional line_profiler)."""

import argparse
import cProfile
import io
import json
import os
import pathlib
import pstats
import random
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml import HAML
from haml.dynamics.attractor import Attractor
from haml.dynamics.integrator import ODEIntegrator
from haml.dynamics.level import Level
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


def load_fashion_mnist_subset(n_train, n_test, seed):
    cache_dir = os.path.join(os.getcwd(), ".sklearn_data")
    os.makedirs(cache_dir, exist_ok=True)
    ds = None
    fetch_errors = []
    # Try by canonical name first.
    try:
        ds = fetch_openml("Fashion-MNIST", version=1, parser="auto", data_home=cache_dir)
    except Exception as exc:
        fetch_errors.append(f"name/version lookup failed: {exc}")
    # Fallback: stable OpenML dataset id for Fashion-MNIST.
    if ds is None:
        try:
            ds = fetch_openml(data_id=40996, parser="auto", data_home=cache_dir)
        except Exception as exc:
            fetch_errors.append(f"data_id lookup failed: {exc}")
    if ds is None:
        details = " | ".join(fetch_errors) if fetch_errors else "unknown OpenML error"
        raise RuntimeError(
            "Unable to load Fashion-MNIST from OpenML. "
            "If this environment blocks OpenML, use '--dataset make_moons' for profiling, "
            "or retry later when OpenML is reachable. "
            f"Details: {details}"
        )
    X = ds.data.to_numpy()
    y = ds.target.to_numpy().astype(int)

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


def load_moons_subset(n_train, n_test, seed, noise):
    X, y = make_moons(n_samples=n_train + n_test, noise=noise, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def build_trainer(model, n_epochs, phase1_epochs, phase2_epochs, phase1_max_steps, phase2_max_steps, phase3_max_steps):
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
            phase1_max_steps=phase1_max_steps,
            phase2_max_steps=phase2_max_steps,
            phase3_max_steps=phase3_max_steps,
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
            chance_tolerance=0.12,
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
        verbose=False,
    )


def run_probe(args):
    phase1_epochs = max(0, min(args.phase1_epochs, args.n_epochs))
    phase2_epochs = max(0, min(args.phase2_epochs, max(0, args.n_epochs - phase1_epochs)))
    alpha_bu, alpha_td = (0.0, 0.0) if args.mode == "independent" else (0.5, 1.5)

    set_seed(args.seed)
    if args.dataset == "fashion_mnist":
        X_train, X_test, y_train, y_test = load_fashion_mnist_subset(args.n_train, args.n_test, args.seed)
        level_dims = None
    else:
        X_train, X_test, y_train, y_test = load_moons_subset(args.n_train, args.n_test, args.seed, args.noise)
        level_dims = [2, 2]

    model = HAML(
        n_levels=3,
        level_dims=level_dims,
        n_attractors_per_class=args.n_attractors_per_class,
        alpha_bu=alpha_bu,
        alpha_td=alpha_td,
        use_vectorized_levels=args.use_vectorized_levels,
        repulsion_mode=args.repulsion_mode,
        max_steps=args.max_steps,
        tol=None if args.disable_convergence_check else args.tol,
        convergence_check_every=args.convergence_check_every,
        mu_dyn=args.mu_dyn,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        train_on_fit=False,
        device=args.device,
    )
    model.fit(X_train, y_train)
    trainer = build_trainer(
        model,
        args.n_epochs,
        phase1_epochs,
        phase2_epochs,
        args.phase1_max_steps,
        args.phase2_max_steps,
        args.phase3_max_steps,
    )

    train_start = time.perf_counter()
    history = trainer.train(X_train, y_train)
    train_time_sec = time.perf_counter() - train_start
    test_acc = model.score(X_test, y_test)

    return {
        "train_time_sec": float(train_time_sec),
        "test_accuracy": float(test_acc),
        "final_train_accuracy": float(history["accuracy"][-1]) if history["accuracy"] else None,
        "n_history_epochs": int(len(history["accuracy"])),
    }


def write_cprofile_stats(profile, out_dir, top_n):
    profile_path = out_dir / "cprofile_stats.prof"
    by_cumtime_path = out_dir / "cprofile_top_cumtime.txt"
    by_tottime_path = out_dir / "cprofile_top_tottime.txt"
    profile.dump_stats(str(profile_path))

    s_cum = io.StringIO()
    stats_cum = pstats.Stats(profile, stream=s_cum).sort_stats("cumtime")
    stats_cum.print_stats(top_n)
    by_cumtime_path.write_text(s_cum.getvalue(), encoding="utf-8")

    s_tot = io.StringIO()
    stats_tot = pstats.Stats(profile, stream=s_tot).sort_stats("tottime")
    stats_tot.print_stats(top_n)
    by_tottime_path.write_text(s_tot.getvalue(), encoding="utf-8")

    return {
        "profile_path": str(profile_path),
        "top_cumtime_path": str(by_cumtime_path),
        "top_tottime_path": str(by_tottime_path),
    }


def run_line_by_line_if_available(args, out_dir):
    try:
        from line_profiler import LineProfiler
    except Exception as exc:
        note_path = out_dir / "line_profiler_unavailable.txt"
        note_path.write_text(f"line_profiler unavailable: {exc}\n", encoding="utf-8")
        return {"enabled": False, "note_path": str(note_path)}

    lp = LineProfiler()
    lp.add_function(HAMLTrainer.train)
    lp.add_function(ODEIntegrator.integrate)
    lp.add_function(ODEIntegrator.step_rk4)
    lp.add_function(ODEIntegrator.compute_forces)
    lp.add_function(Level.intra_level_force)
    lp.add_function(Level.predict_class_scores)
    lp.add_function(Attractor.attraction_force)
    lp.add_function(Attractor.repulsion_force)
    lp.add_function(Attractor.attraction_strength)

    profiled_run = lp(run_probe)
    profiled_run(args)

    out_path = out_dir / "line_profiler_report.txt"
    with out_path.open("w", encoding="utf-8") as f:
        lp.print_stats(stream=f)

    return {"enabled": True, "report_path": str(out_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime profiling for HAML training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["fashion_mnist", "make_moons"], default="make_moons")
    parser.add_argument("--noise", type=float, default=0.30, help="Noise used for make_moons dataset.")
    parser.add_argument("--mode", choices=["coupled_tuned", "independent"], default="coupled_tuned")
    parser.add_argument("--use-vectorized-levels", action="store_true")
    parser.add_argument("--repulsion-mode", choices=["global", "inter_class_only"], default="global")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--phase1-epochs", type=int, default=1)
    parser.add_argument("--phase2-epochs", type=int, default=0)
    parser.add_argument("--phase1-max-steps", type=int, default=50)
    parser.add_argument("--phase2-max-steps", type=int, default=80)
    parser.add_argument("--phase3-max-steps", type=int, default=100)
    parser.add_argument("--n-attractors-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--convergence-check-every", type=int, default=5)
    parser.add_argument("--disable-convergence-check", action="store_true")
    parser.add_argument("--mu-dyn", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-n", type=int, default=40, help="Top functions to print in cProfile summaries.")
    parser.add_argument("--skip-line-profiler", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/profile_runtime",
        help="Output directory for profiling artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    probe_summary = run_probe(args)
    profiler.disable()

    cprofile_paths = write_cprofile_stats(profiler, out_dir, args.top_n)
    line_profile = {"enabled": False, "skipped": True} if args.skip_line_profiler else run_line_by_line_if_available(args, out_dir)

    summary = {
        "config": vars(args),
        "probe": probe_summary,
        "artifacts": {
            **cprofile_paths,
            "line_profile": line_profile,
        },
        "generated_at_epoch_sec": time.time(),
    }
    summary_path = out_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
