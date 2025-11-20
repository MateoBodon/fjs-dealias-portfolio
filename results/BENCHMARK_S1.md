# Synthetic Benchmark (S1) — Hetzner AX102

- Command: `python experiments/synthetic_oneway/run.py` (default config)
- Runtime: ~2217s wall clock
- Output: `figures/synthetic/summary.json`
- Environment: Python 3.11.14 (conda `fjs`), 16 vCPUs (`nproc`=16)

## Config snapshot
- n_assets=60, n_groups=60, replicates=3, noise_variance=1.0, signal_to_noise=0.35, spike_strength=6.0
- Monte Carlo trials: S1/S3=200, guardrail_trials=200, multi_spike_trials=120
- Other: delta=0.05, delta_frac=0.02, eps=0.03, stability_eta_deg=0.4, a_grid=120

## Bias Reduction (MANOVA → De-aliased eigenvalues)
Computed as `aliased_bias - dealiased_bias` from `summary.json` S3 results.

| µ (true spike) | Aliased bias | De-aliased bias | Bias reduction |
| --- | --- | --- | --- |
| 4.0 | 10.1255 | 0.6902 | 9.4353 |
| 6.0 | 14.1937 | 0.7134 | 13.4803 |
| 8.0 | 17.8687 | 0.6053 | 17.2635 |

Notes:
- Guardrail check: default setting yielded 0/200 false positives; relaxed setting (delta=0, no stability) hit 200/200.
- Multi-spike pairing results and visualizations are in `figures/synthetic/` for reproducibility.
