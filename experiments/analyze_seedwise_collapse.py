"""
Mechanistic seed-wise diagnostics for HAML coupling collapse analysis.

Reads a JSON produced by experiments/seedwise_coupling_diagnostics.py and
prints a compact report to identify level-wise degraded runs.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze seed-wise collapse traces.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="experiments/diag_seed42_43_mechanistic.json",
    )
    parser.add_argument(
        "--phase3-start-epoch",
        type=int,
        default=14,
        help="1-based epoch index where phase 3 starts.",
    )
    parser.add_argument(
        "--chance-threshold",
        type=float,
        default=0.52,
        help="Level accuracy <= threshold is considered chance-like.",
    )
    parser.add_argument(
        "--sep-threshold",
        type=float,
        default=0.08,
        help="Separation ratio threshold used to flag potential collapse.",
    )
    return parser.parse_args()


def _max_chance_streak(values, chance_threshold):
    streak = 0
    best = 0
    for v in values:
        if v <= chance_threshold:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _epoch_slice(values, start_epoch_1b):
    idx = max(0, start_epoch_1b - 1)
    return values[idx:]


def analyze_run(run, phase3_start_epoch, chance_threshold, sep_threshold):
    history = run["history"]
    n_epochs = len(history["accuracy"])
    n_levels = len(history["level_accuracy"][0])
    phase3_idx = max(0, phase3_start_epoch - 1)

    print(f"\n=== Seed {run['seed']} | test_acc={run['test_accuracy']:.3f} ===")
    print(
        "epoch,count:"
        f" n_epochs={n_epochs}, phase3_start={phase3_start_epoch}, "
        f"chance_threshold={chance_threshold:.2f}, sep_threshold={sep_threshold:.2f}"
    )

    flagged_levels = []
    for level_idx in range(n_levels):
        level_acc = [ep[level_idx] for ep in history["level_accuracy"]]
        level_diag = [ep[level_idx] for ep in history["level_attractor_diagnostics"]]
        sep_ratio = [d["separation_ratio"] for d in level_diag]

        phase3_acc = _epoch_slice(level_acc, phase3_start_epoch)
        phase3_sep = _epoch_slice(sep_ratio, phase3_start_epoch)

        max_streak = _max_chance_streak(phase3_acc, chance_threshold)
        min_sep = min(phase3_sep) if phase3_sep else min(sep_ratio)
        final_acc = level_acc[-1]
        final_sep = sep_ratio[-1]

        chance_locked = (max_streak >= 3) and (final_acc <= chance_threshold)
        separation_collapse = min_sep < sep_threshold
        is_flagged = chance_locked or separation_collapse
        if is_flagged:
            flagged_levels.append(level_idx)

        print(
            f"  L{level_idx}: final_acc={final_acc:.3f}, final_sep={final_sep:.4f}, "
            f"phase3_min_sep={min_sep:.4f}, phase3_max_chance_streak={max_streak}, "
            f"chance_locked={chance_locked}, sep_collapse={separation_collapse}, flagged={is_flagged}"
        )

    if not flagged_levels:
        print("  No level met collapse criteria.")
        return

    print("  Timeline (flagged levels):")
    print("    epoch | level | level_acc | sep_ratio | inter_min | intra_mean | mu_sep | lr | div")

    for level_idx in flagged_levels:
        for e in range(phase3_idx, n_epochs):
            d = history["level_attractor_diagnostics"][e][level_idx]
            print(
                f"    {e + 1:>5} | L{level_idx} | "
                f"{history['level_accuracy'][e][level_idx]:.3f} | "
                f"{d['separation_ratio']:.4f} | "
                f"{d['inter_centroid_dist_min']:.4f} | "
                f"{d['intra_spread_mean']:.4f} | "
                f"{history['mu_sep'][e]:.4f} | "
                f"{history['lr'][e]:.6f} | "
                f"{history['level_divergence'][e]:.4f}"
            )


def main():
    args = parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("No runs found in input JSON.")

    print(f"Input: {args.input_json}")
    for run in runs:
        analyze_run(
            run=run,
            phase3_start_epoch=args.phase3_start_epoch,
            chance_threshold=args.chance_threshold,
            sep_threshold=args.sep_threshold,
        )


if __name__ == "__main__":
    main()
