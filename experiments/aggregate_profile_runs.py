"""Aggregate profile_runtime results across multiple run directories."""

import argparse
import json
import pathlib
import statistics


def _load_summary(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    probe = data.get("probe", {})
    cfg = data.get("config", {})
    return {
        "path": str(path),
        "seed": cfg.get("seed"),
        "train_time_sec": float(probe.get("train_time_sec")),
        "test_accuracy": float(probe.get("test_accuracy")),
        "final_train_accuracy": float(probe.get("final_train_accuracy")),
        "resolved_device": probe.get("resolved_device", cfg.get("device")),
    }


def _stats(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def main():
    parser = argparse.ArgumentParser(description="Aggregate profile_summary.json over multiple run directories.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="List of directories containing profile_summary.json",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional output json path.",
    )
    args = parser.parse_args()

    runs = []
    for run_dir in args.run_dirs:
        p = pathlib.Path(run_dir) / "profile_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        runs.append(_load_summary(p))

    train_times = [r["train_time_sec"] for r in runs]
    test_accs = [r["test_accuracy"] for r in runs]
    train_accs = [r["final_train_accuracy"] for r in runs]

    t_mean, t_std = _stats(train_times)
    te_mean, te_std = _stats(test_accs)
    tr_mean, tr_std = _stats(train_accs)

    summary = {
        "n_runs": len(runs),
        "seeds": [r["seed"] for r in runs],
        "resolved_devices": sorted(set(r["resolved_device"] for r in runs)),
        "train_time_sec_mean": t_mean,
        "train_time_sec_std": t_std,
        "test_accuracy_mean": te_mean,
        "test_accuracy_std": te_std,
        "final_train_accuracy_mean": tr_mean,
        "final_train_accuracy_std": tr_std,
        "runs": runs,
    }

    if args.out_json:
        out_path = pathlib.Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

