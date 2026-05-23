"""Ablation runner for model-quality levers B1/B2/B3/B23.

B1: repulsion_mode (global vs inter_class_only)
B2: sigma_init_mode (sqrt_d_std vs median_pairwise)
B3: learn_projections (False vs True)
B23: sigma_init_mode x learn_projections (2x2)
"""

import argparse
import json
import pathlib
import random
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml import HAML
from haml.training import ConstrainedOptimizer, HAMLLoss, HAMLTrainer, PhaseConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(name, n_train, n_test, seed, noise):
    if name == "make_moons":
        X, y = make_moons(n_samples=n_train + n_test, noise=noise, random_state=seed)
    else:
        cache_dir = pathlib.Path.cwd() / ".sklearn_data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds = fetch_openml(data_id=40996, parser="auto", data_home=str(cache_dir))
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


def build_model_overrides(args, variant_overrides):
    """Compose model overrides with a frozen Fashion-MNIST CPU reference."""
    resolved = {}
    if args.dataset == "fashion_mnist" and str(args.device).lower() == "cpu":
        # Reference config validated by B23:
        # sigma_init_mode='sqrt_d_std' + learn_projections=True
        resolved.update(
            {
                "sigma_init_mode": "sqrt_d_std",
                "learn_projections": True,
            }
        )
    resolved.update(variant_overrides)
    return resolved


def run_single(args, seed, overrides):
    set_seed(seed)
    X_train, X_test, y_train, y_test = load_dataset(
        args.dataset, args.n_train, args.n_test, seed, args.noise
    )

    level_dims = None if args.dataset == "fashion_mnist" else [2, 2]
    model_overrides = build_model_overrides(args, overrides)

    model = HAML(
        n_levels=3,
        level_dims=level_dims,
        n_attractors_per_class=args.n_attractors_per_class,
        alpha_bu=0.5,
        alpha_td=1.5,
        use_vectorized_levels=True,
        training_preset=args.training_preset,
        convergence_check_every=args.convergence_check_every,
        mu_dyn=0.0,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        train_on_fit=False,
        device=args.device,
        **model_overrides,
    )
    model.fit(X_train, y_train)

    trainer = HAMLTrainer(
        model=model,
        optimizer=ConstrainedOptimizer(model.parameters(), lr=model.lr),
        loss_fn=HAMLLoss(mu_sep=model.mu_sep, mu_dyn=model.mu_dyn),
        phase_config=PhaseConfig(
            n_epochs=args.n_epochs,
            batch_size=model.batch_size,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase1_max_steps=model.phase1_max_steps,
            phase2_max_steps=model.phase2_max_steps,
            phase3_max_steps=model.phase3_max_steps,
            phase1_integrator_method=model.phase1_integrator_method,
            phase2_integrator_method=model.phase2_integrator_method,
            phase3_integrator_method=model.phase3_integrator_method,
            diagnostics_every_epochs=1,
            diagnostics_subset_size=None,
        ),
        device=model.device,
        verbose=False,
    )

    t0 = time.perf_counter()
    history = trainer.train(X_train, y_train)
    dt = time.perf_counter() - t0
    test_acc = model.score(X_test, y_test)
    return {
        "seed": seed,
        "train_time_sec": float(dt),
        "test_accuracy": float(test_acc),
        "final_train_accuracy": float(history["accuracy"][-1]),
    }


def aggregate(rows):
    train = [r["train_time_sec"] for r in rows]
    test = [r["test_accuracy"] for r in rows]
    tr = [r["final_train_accuracy"] for r in rows]
    return {
        "n_runs": len(rows),
        "train_time_sec_mean": float(np.mean(train)),
        "train_time_sec_std": float(np.std(train)),
        "test_accuracy_mean": float(np.mean(test)),
        "test_accuracy_std": float(np.std(test)),
        "final_train_accuracy_mean": float(np.mean(tr)),
        "final_train_accuracy_std": float(np.std(tr)),
    }


def _atomic_write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _base_report(args):
    return {
        "ablation": args.ablation,
        "dataset": args.dataset,
        "seeds": args.seeds,
        "config": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_epochs": args.n_epochs,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "training_preset": args.training_preset,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "variants": {},
    }


def _load_resume_report(args, expected_variants):
    out = pathlib.Path(args.out_json)
    if (not args.resume) or (not out.exists()):
        return _base_report(args)

    try:
        existing = json.loads(out.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unable to read resume file '{out}': {exc}") from exc

    expected = _base_report(args)
    if (
        existing.get("ablation") != expected["ablation"]
        or existing.get("dataset") != expected["dataset"]
        or existing.get("seeds") != expected["seeds"]
        or existing.get("config") != expected["config"]
    ):
        raise RuntimeError(
            "Resume file exists but command/config does not match. "
            "Use a new --out-json file or run with --resume disabled."
        )

    variants = existing.get("variants", {})
    for vname, _ in expected_variants:
        if vname not in variants:
            variants[vname] = {"runs": [], "summary": {}, "overrides": {}}
        variants[vname]["runs"] = variants[vname].get("runs", [])
    existing["variants"] = variants
    return existing


def build_variants(ablation):
    if ablation == "b1":
        return [
            ("repulsion_global", {"repulsion_mode": "global"}),
            ("repulsion_inter_class_only", {"repulsion_mode": "inter_class_only"}),
        ]
    if ablation == "b2":
        return [
            ("sigma_sqrt_d_std", {"sigma_init_mode": "sqrt_d_std"}),
            ("sigma_median_pairwise", {"sigma_init_mode": "median_pairwise"}),
        ]
    if ablation == "b3":
        return [
            ("proj_fixed", {"learn_projections": False}),
            ("proj_learned", {"learn_projections": True}),
        ]
    if ablation == "b23":
        return [
            (
                "sigma_sqrt_d_std_proj_fixed",
                {"sigma_init_mode": "sqrt_d_std", "learn_projections": False},
            ),
            (
                "sigma_sqrt_d_std_proj_learned",
                {"sigma_init_mode": "sqrt_d_std", "learn_projections": True},
            ),
            (
                "sigma_median_pairwise_proj_fixed",
                {"sigma_init_mode": "median_pairwise", "learn_projections": False},
            ),
            (
                "sigma_median_pairwise_proj_learned",
                {"sigma_init_mode": "median_pairwise", "learn_projections": True},
            ),
        ]
    raise ValueError(f"Unknown ablation: {ablation}")


def parse_args():
    p = argparse.ArgumentParser(description="Run B1/B2/B3/B23 ablations with consistent protocol.")
    p.add_argument("--ablation", choices=["b1", "b2", "b3", "b23"], required=True)
    p.add_argument("--dataset", choices=["fashion_mnist", "make_moons"], default="fashion_mnist")
    p.add_argument("--noise", type=float, default=0.30)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--phase1-epochs", type=int, default=3)
    p.add_argument("--phase2-epochs", type=int, default=3)
    p.add_argument(
        "--training-preset",
        choices=[
            "fast_train_cpu",
            "ultra_fast_train_cpu",
            "fashion_cpu_accuracy",
            "fashion_cpu_runtime",
        ],
        default="ultra_fast_train_cpu",
    )
    p.add_argument("--n-attractors-per-class", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--convergence-check-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --out-json and only run missing seeds.",
    )
    p.add_argument("--out-json", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    variants = build_variants(args.ablation)
    report = _load_resume_report(args, variants)
    for name, overrides in variants:
        variant = report["variants"].setdefault(
            name, {"runs": [], "summary": {}, "overrides": overrides}
        )
        variant["overrides"] = overrides
        existing_runs = variant.get("runs", [])
        done_seeds = {int(r["seed"]) for r in existing_runs}

        for seed in args.seeds:
            if int(seed) in done_seeds:
                continue
            row = run_single(args, seed, overrides)
            existing_runs.append(row)
            done_seeds.add(int(seed))
            variant["runs"] = sorted(existing_runs, key=lambda r: int(r["seed"]))
            variant["summary"] = aggregate(variant["runs"])
            _atomic_write_json(args.out_json, report)

        variant["runs"] = sorted(existing_runs, key=lambda r: int(r["seed"]))
        variant["summary"] = aggregate(variant["runs"])

    _atomic_write_json(args.out_json, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
