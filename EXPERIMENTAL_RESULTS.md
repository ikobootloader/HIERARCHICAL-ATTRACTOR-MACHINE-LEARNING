# Experimental Results - HAML vs SA-nODE (Reimplementation)

## Scope
This document reports the controlled experimental comparison between:
- HAML (coupled hierarchical dynamics)
- SA-nODE-reimpl (flat-state reimplementation inspired by Marino et al.)
- Baselines KNN and SVM-RBF (for contextual reference)

The objective is to characterize both absolute performance and robustness under increasing noise.

## Protocol
- Dataset: `make_moons`
- Samples: `3000`
- Train/Test split: `75/25`
- Seed: `42`
- Noise levels: `0.10`, `0.20`, `0.30`, `0.40`
- Preprocessing: `StandardScaler` fitted on train only
- Metric: test accuracy

HAML configuration (fixed):
- Coupled: `alpha_bu=0.5`, `alpha_td=1.5`
- Independent reference: `alpha_bu=0.0`, `alpha_td=0.0`

SA-nODE-reimpl architectural constraints (fixed):
- Flat state space (input dimension)
- Planted binary attractors `+-a`
- Analytical double-well dynamics
- No hierarchy

## SA-nODE Hyperparameter Search (Seed Fixed)
Search was performed only on dynamic hyperparameters, without changing architecture:
- `a in {0.8, 1.0, 1.2}`
- `dt in {0.03, 0.05, 0.08}`
- `gamma in {0.0, 0.05, 0.1}`

Selection criteria:
- Primary: maximize mean accuracy across all noise levels
- Secondary: maximize accuracy at noise `0.40`

Selected config (primary):
- `a=1.0`, `dt=0.03`, `gamma=0.0`

Best config for noise `0.40` (secondary):
- `a=0.8`, `dt=0.03`, `gamma=0.1`

Observed gain at `0.40` from secondary config is limited; primary config is retained.

## Consolidated Results
| Noise | HAML Coupled | SA-nODE Tuned | Delta (HAML - SA-nODE) |
|---|---:|---:|---:|
| 0.10 | 99.60% | 86.67% | +12.93 pts |
| 0.20 | 95.60% | 86.40% | +9.20 pts |
| 0.30 | 90.53% | 85.87% | +4.67 pts |
| 0.40 | 85.07% | 83.33% | +1.73 pts |

Degradation from noise `0.10` to `0.40`:
- HAML coupled: `-14.53 pts`
- SA-nODE tuned: `-3.33 pts`

## Main Findings
1. Absolute level:
- HAML coupled outperforms SA-nODE tuned at every tested noise level.
- The margin decreases with noise (large at low/intermediate noise, marginal at high noise).

2. Robustness slope:
- SA-nODE tuned degrades less with increasing noise.
- HAML achieves higher absolute accuracy but is more sensitive to strong noise.

3. Interpretation:
- The hierarchical mechanism provides substantial gains when signal structure remains exploitable.
- Under stronger corruption, PCA projections capture noise structure rather than class structure, which weakens hierarchical signal quality.
- SA-nODE operates directly in raw input space and is less exposed to projection-induced sensitivity.

## Coupling Ablation Inside HAML
To isolate the effect of bidirectional coupling (and not only the broader HAML architecture), we compare independent vs coupled HAML under the same protocol.

| Noise | HAML Independent | HAML Coupled | Coupling Delta |
|---|---:|---:|---:|
| 0.10 | 99.60% | 99.60% | +0.00 pts |
| 0.20 | 89.33% | 95.60% | +6.27 pts |
| 0.30 | 88.13% | 90.53% | +2.40 pts |
| 0.40 | 84.67% | 85.07% | +0.40 pts |

This confirms the core pattern expected from the theory:
- negligible gain in an easy regime (low noise),
- maximal gain in an intermediate ambiguity regime,
- small gain in a heavily corrupted regime.

## Valid Claims
- In a controlled implementation setting, hierarchical bidirectional dynamics improve absolute performance over a flat SA-nODE-style model across tested noise levels.
- At high noise (`0.40`), the margin is small (`+1.73 pts`) and should be treated as fragile in single-seed analysis.
- Bidirectional coupling provides a measurable gain over independent HAML in intermediate noise regimes (`+6.27 pts` at noise `0.20`), with negligible effect at low and high noise, consistent with the theoretical prediction of top-down guidance under local ambiguity.
- Adjoint integration preserves predictive accuracy while reducing peak traced memory by about `147x`, at a runtime cost of about `1.87x` on CPU; this supports adjoint as an enabler for memory-constrained configurations rather than a general speed optimization.

## Limits
- SA-nODE benchmark is a paper-inspired internal reimplementation, not the authors' official codebase.
- Multi-seed confidence intervals are required before strong claims in the high-noise regime.

## Reproducibility
- Main comparison script: `experiments/noisy_sanode_comparison.py`
- SA-nODE tuning script: `experiments/tune_sanode_reimpl.py`
- Outputs:
  - `experiments/noisy_sanode_comparison_output.txt`
  - `experiments/tune_sanode_reimpl_output.txt`
  - `experiments/noisy_sanode_comparison_tuned_summary.json`

## Integrator Benchmark: RK4 vs Adjoint
To validate the new adjoint path (`torchdiffeq`) on the same HAML setup, we ran a controlled benchmark with fixed seed and identical training configuration.

Configuration:
- Dataset: `make_moons`, `n_samples=1800`, `noise=0.25`
- Seed: `42`
- Train/Test: `75/25`
- HAML: coupled (`alpha_bu=0.5`, `alpha_td=1.5`)
- Training: `5` epochs, batch size `64`, `max_steps=35`, `dt=0.1`

Results:
| Method | Accuracy | Elapsed Time | Peak tracemalloc |
|---|---:|---:|---:|
| RK4 | 87.33% | 190.50 s | 66.16 MB |
| Adjoint | 87.33% | 356.71 s | 0.45 MB |

Relative:
- Time ratio (Adjoint / RK4): `1.87x`
- Accuracy delta (Adjoint - RK4): `0.00 pt`
- Peak tracemalloc delta (Adjoint - RK4): `-65.71 MB`

Interpretation:
- On this CPU setup, adjoint preserved accuracy while substantially reducing traced peak memory.
- The memory gain came with a significant runtime cost.
- This is consistent with adjoint trade-offs (memory efficiency vs compute overhead), and motivates method selection by resource constraints.
- In the current regime (short trajectories on CPU), RK4 is faster; adjoint should be viewed as an enabler for memory-constrained regimes rather than a speed optimization here.

Practical implication:
- Adjoint is primarily useful to unlock configurations that may be inaccessible with direct RK4 (longer trajectories, larger batch sizes, tighter memory budgets), while preserving predictive performance.

Current limitations of this benchmark:
- The time crossover point (trajectory length / batch size where adjoint becomes faster) was not measured yet.
- The benchmark was run on CPU only; memory/time trade-offs can differ on GPU, where memory pressure is typically the dominant bottleneck.
