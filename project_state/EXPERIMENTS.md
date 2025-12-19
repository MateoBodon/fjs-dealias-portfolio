---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Experiments & Configs

- **Equity panel (weekly designs)** — `experiments/equity_panel/run.py`
  - Configs: `config.yaml` (full RC), `config.rc.yaml` (gallery/memo), `config.smoke.yaml` (fast smoke), `config.nested.smoke.yaml`, `config.ablation.smoke.yaml`, `config.gallery.yaml`, crisis presets (`config.crisis.2020.yaml`, `config.crisis.2022.yaml`, `config.nested.crisis.2020.yaml`).
  - CLI overrides: design (`dow`/`vol`/`nested`/`oneway`), estimator (`dealias`, `lw`, `oas`, `cc`, `factor`, `tyler_shrink`), edge-mode (`scm`/`tyler`/`huber`), gating (`--gating-mode`, `--gating-calibration`, `--gating-diagnostics`), prewhiten/factors, minvar controls (`--minvar-ridge`, `--minvar-box`, `--minvar-condition-cap`, `--turnover-cost`), crisis window, cache/resume flags, ablation switches.
  - Outputs: detection summaries, spectra/variance/VaR plots, gating diagnostics, `config_resolved.yaml`, `panel_manifest.json`, cached per-window stats in `.cache/` when resume/precompute enabled.
- **Daily evaluation (rc/rc-lite/rc-lite-sanity)** — `experiments/eval/run.py`
  - Config defaults in `experiments/eval/config.yaml` + thresholds `thresholds.json`; CLI merges via `experiments/eval/config.py`.
  - Key knobs: group design (`dow`, `vol`, `week`, `dowxvol`), overlay (`overlay_delta`, `overlay_delta_frac`, `q_max`, `gate_mode`, `gate_delta_frac_min/max`, `gate_stability_min`, `gate_accept_nonisolated`, `coarse_candidate`), edge (`edge_mode`), MV (`mv_gamma`, `mv_tau`, `mv_box_lo/hi`, `mv_turnover_bps`, `mv_condition_cap`, `mv_solver {projgrad,cvxpy}`, `mv_skip_on_missing_solver`, `mv_solver_name`), prewhitening (`prewhiten`, `use_factor_prewhiten`), isolation (`require_isolated`, `q2_alignment_min_cos`), bootstrap samples.
  - Typical commands: `make rc-dow`, `make rc-vol`, `make rc-week`, `make rc-dowxvol`, or manual `python -m experiments.eval.run --returns-csv data/returns_daily.csv --group-design dow --window 126 --horizon 21 --out reports/rc-<date>/dow-tyler ...`.
- **Synthetic harness** — `experiments/synthetic/null.py`, `power.py`, `power_null.py`, `nested_killtest.py`, `calibrate_thresholds.py`.
  - Use `HARNESS_TRIALS` and `EXEC_MODE` envs; configs for nested kill-test in `config.nested.killtest.yaml`.
  - Outputs live in `reports/synthetic/` + `reports/figures/`; calibration JSONs in `calibration/`.
- **Ablations** — `experiments/ablate/run.py` with `ablation_matrix.yaml` or `ablation_matrix_tiny.yaml`; `make rc-ablations` runs after rc/rc-lite to summarise ablation outputs in `experiments/equity_panel/outputs_ablation_*`.
- **Prewhitening / ETF / daily smoke**
  - `experiments/prewhiten.py` (standalone prewhitening), `experiments/etf_panel/run.py` (ETF demo), `experiments/daily/run.py` (quick DoW/vol smoke via Make `smoke-daily`).
- **Synthetic-oneway demo** — `experiments/synthetic_oneway/run.py` for S1/S3/S4/S5 figures; invoked by `make figures`.
- **Outputs to avoid overwriting** — keep `reports/rc-YYYYMMDD/`, `experiments/equity_panel/outputs_*`, `reports/rc-sensitivity/`, `reports/aws/` immutable; create new timestamped dirs for reruns.
