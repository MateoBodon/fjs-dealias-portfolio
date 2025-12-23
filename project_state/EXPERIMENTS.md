---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Experiments & Configs

- **Equity panel (weekly designs)** — `experiments/equity_panel/run.py`
  - Configs: `config.yaml` (full RC), `config.rc.yaml` (gallery/memo), `config.smoke.yaml`, `config.nested.smoke.yaml`, `config.nested.smoke.tiny.yaml`, `config.ablation.smoke.yaml`, `config.gallery.yaml`, crisis presets (`config.crisis.2020.yaml`, `config.crisis.2022.yaml`, `config.nested.crisis.2020.yaml`).
  - CLI overrides: design (`dow`/`vol`/`nested`/`oneway`), estimator (`dealias`, `lw`, `oas`, `cc`, `factor`, `tyler_shrink`), edge-mode (`scm`/`tyler`/`huber`), gating (`--gating-mode`, `--gating-calibration`, `--gating-diagnostics`), prewhiten/factors, minvar controls (`--minvar-ridge`, `--minvar-box`, `--minvar-condition-cap`, `--turnover-cost`), cache/resume flags, ablation switches, `--max-windows` cap.
  - Outputs: detection summaries, spectra/variance/VaR plots, gating diagnostics (optional), `config_resolved.yaml`, `panel_manifest.json`, cached per-window stats in `.cache/`.
- **Daily evaluation (rc/rc-lite/rc-lite-sanity)** — `experiments/eval/run.py`
  - Defaults in `experiments/eval/config.yaml` + `experiments/eval/thresholds.json` via `experiments/eval/config.py`.
  - Key knobs: group design (`dow`, `vol`, `week`, `dowxvol`), overlay/gating (`overlay_delta`, `q_max`, `gate_mode`, `gate_delta_frac_min/max`, `gate_stability_min`, `gate_accept_nonisolated`, `coarse_candidate`), edge (`edge_mode`), MV (`mv_gamma`, `mv_tau`, `mv_box_lo/hi`, `mv_turnover_bps`, `mv_condition_cap`, `mv_solver`, `mv_skip_on_missing_solver`, `mv_solver_name`), prewhitening (`prewhiten`, `use_factor_prewhiten`), comparison validity (`min_comparison_windows`).
  - Typical commands: `make rc-dow`, `make rc-vol`, `make rc-week`, `make rc-dowxvol`, or `python -m experiments.eval.run --returns-csv data/returns_daily.csv --group-design dow --window 126 --horizon 21 --out reports/rc-<date>/dow-tyler ...`.
- **Synthetic harness** — `experiments/synthetic/null.py`, `power.py`, `power_null.py`, `nested_killtest.py`, `calibrate_thresholds.py`.
  - Configs: `experiments/synthetic/config.nested.killtest.yaml` (nested kill-test).
  - Outputs: `reports/synthetic/` + `calibration/*.json` + `reports/figures/*`.
- **Ablations** — `experiments/ablate/run.py` with `ablation_matrix.yaml` / `ablation_matrix_tiny.yaml`; `make rc-ablations` summarizes into `experiments/equity_panel/outputs_ablation_*`.
- **Prewhitening / ETF / daily smoke** — `experiments/prewhiten.py`, `experiments/etf_panel/run.py`, `experiments/daily/run.py` (Make target `smoke-daily`).
- **Synthetic-oneway demo** — `experiments/synthetic_oneway/run.py` for S1/S3/S4/S5 figures (Make target `figures`).
- **Outputs to avoid overwriting** — keep `reports/rc-YYYYMMDD/`, `experiments/equity_panel/outputs_*`, `reports/rc-sensitivity/`, `reports/aws/` immutable; create new timestamped dirs for reruns.
