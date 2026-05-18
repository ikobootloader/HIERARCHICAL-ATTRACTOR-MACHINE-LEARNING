"""Micro-benchmark: Level vs LevelVectorized (forward and backward)."""

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haml.dynamics.level import Level
from haml.dynamics.level_vectorized import LevelVectorized


def sync_if_cuda(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build_pair(device, batch_size, dim, n_classes, n_attr, seed):
    torch.manual_seed(seed)
    level_ref = Level(
        level_idx=0,
        dim=dim,
        n_classes=n_classes,
        n_attractors_per_class=n_attr,
        lambda_repulsion=0.5,
        rho_sigma_ratio=2.0,
    ).to(device)
    level_vec = LevelVectorized(
        level_idx=0,
        dim=dim,
        n_classes=n_classes,
        n_attractors_per_class=n_attr,
        lambda_repulsion=0.5,
        rho_sigma_ratio=2.0,
        repulsion_mode="global",
    ).to(device)

    x_init = torch.randn(batch_size * 4, dim, device=device)
    y_init = torch.randint(0, n_classes, (batch_size * 4,), device=device)
    y_init[:n_classes] = torch.arange(n_classes, device=device)
    level_ref.initialize_attractors(x_init, y_init)

    with torch.no_grad():
        for c in range(n_classes):
            for m, attractor in enumerate(level_ref.attractors[str(c)]):
                idx = c * n_attr + m
                level_vec.positions[idx].copy_(attractor.position)
                level_vec.log_sigma[idx].copy_(attractor.log_sigma)
                level_vec.log_rho[idx].copy_(attractor.log_rho)
                level_vec.log_weight[idx].copy_(attractor.log_weight)

    x = torch.randn(batch_size, dim, device=device)
    return level_ref, level_vec, x


def benchmark_forward(level, x, iters, warmup, fn_name):
    fn = getattr(level, fn_name)
    for _ in range(warmup):
        _ = fn(x)
    sync_if_cuda(x.device.type)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(x)
    sync_if_cuda(x.device.type)
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def benchmark_forward_backward(level, x, iters, warmup, fn_name):
    fn = getattr(level, fn_name)
    for _ in range(warmup):
        xw = x.clone().detach().requires_grad_(True)
        out = fn(xw).sum()
        out.backward()
    sync_if_cuda(x.device.type)
    t0 = time.perf_counter()
    for _ in range(iters):
        xb = x.clone().detach().requires_grad_(True)
        out = fn(xb).sum()
        out.backward()
    sync_if_cuda(x.device.type)
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark vectorized level kernels.")
    parser.add_argument("--batch-size", type=int, default=125)
    parser.add_argument("--dim", type=int, default=784)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-attractors", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    level_ref, level_vec, x = build_pair(
        device=device,
        batch_size=args.batch_size,
        dim=args.dim,
        n_classes=args.n_classes,
        n_attr=args.n_attractors,
        seed=args.seed,
    )

    result = {"device": device, "config": vars(args), "timings_sec": {}}
    for fn_name in ("intra_level_force", "predict_class_scores"):
        ref_fwd = benchmark_forward(level_ref, x, args.iters, args.warmup, fn_name)
        vec_fwd = benchmark_forward(level_vec, x, args.iters, args.warmup, fn_name)
        ref_bwd = benchmark_forward_backward(level_ref, x, args.iters, args.warmup, fn_name)
        vec_bwd = benchmark_forward_backward(level_vec, x, args.iters, args.warmup, fn_name)
        result["timings_sec"][fn_name] = {
            "reference_forward": ref_fwd,
            "vectorized_forward": vec_fwd,
            "speedup_forward": ref_fwd / max(vec_fwd, 1e-12),
            "reference_forward_backward": ref_bwd,
            "vectorized_forward_backward": vec_bwd,
            "speedup_forward_backward": ref_bwd / max(vec_bwd, 1e-12),
        }

    if args.out_json:
        out_path = pathlib.Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
