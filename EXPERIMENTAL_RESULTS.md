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
- Under stronger corruption, hierarchy/projection dependence becomes a sensitivity factor.

## Valid Claims
- In a controlled implementation setting, hierarchical bidirectional dynamics improve absolute performance over a flat SA-nODE-style model across tested noise levels.
- At high noise (`0.40`), the margin is small (`+1.73 pts`) and should be treated as fragile in single-seed analysis.

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
