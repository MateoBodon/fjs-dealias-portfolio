---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Pipeline Flow

- **RC / RC-lite (Makefile)**
  - `make rc` → `rc-data` (equity_panel smoke + crisis runs across estimators) → `rc-eval` (daily evaluation) → `rc-ablations` (ablation grid) → `rc-summary` (`tools/make_summary.py`) → `memo` (gallery + memo). Outputs under `reports/rc-<YYYYMMDD>/` and `experiments/equity_panel/outputs_*`.
  - `make rc-lite` trims to smoke + crisis (dealias/lw/oas) then refreshes gallery/memo.
  - `make rc-lite-sanity` (deterministic DoW+vol slice, top-50 assets, Jan–Jun 2023) runs `experiments/eval/run.py` twice (dow/vol), weekly DoW + nested smoke via `experiments/equity_panel/run.py`, then `tools/make_summary.py` + `tools/summarize_rc_sanity.py`. Outputs in `reports/rc-<date>-sanity-<stamp>/` plus `experiments/equity_panel/outputs_rc-lite-<date>_<stamp>/`.
  - Remote variants: `make aws:<target>` dispatch through `scripts/aws_run.sh` with `AWS_ARGS`.
- **Daily evaluation runner** (`python -m experiments.eval.run`)
  - Inputs: `--returns-csv`, optional `--factors-csv`, group design (`dow`, `vol`, `week`, `dowxvol`), overlay/gate knobs (`--overlay-delta`, `--gate-mode`, `--gate-delta-frac-min`, `--q-max`, `--edge-mode`, `--coarse-candidate`, `--require-isolated`), MV solver controls (`--mv-solver {projgrad,cvxpy}`, `--mv-skip-on-missing-solver`), prewhitening (`--prewhiten`, `--use-factor-prewhiten`).
  - Outputs: metrics/risk/dm CSVs, `diagnostics.csv` + `diagnostics_detail.csv`, plots (histograms when matplotlib present), `resolved_config.json`, `run.json` under `--out`.
- **Weekly equity runner** (`python -m experiments.equity_panel.run`)
  - Inputs: YAML config (see CONFIG_REFERENCE) + CLI overrides (design, estimator, edge-mode, gating-mode/calibration, prewhiten, cache/resume, minvar params, crisis window, ablations). Optional `--gating-diagnostics` emits per-window guardrail detail.
  - Outputs: `detection_summary.csv`, `weekly_diagnostics.md`, spectra and variance/VaR plots, `panel_manifest.json`, `resolved_config.yaml`, cached per-window stats in `.cache/` when `--resume/--precompute-panel` set.
- **Synthetic harness**
  - `make sweep:acceptance` runs `experiments/synthetic/null.py` + `power.py`, writing `reports/synthetic/{null_harness,power_harness}/` and ROC figures under `reports/figures/`; default trials=400 (override `HARNESS_TRIALS`).
  - `make calibrate-thresholds` / `python experiments/synthetic/calibrate_thresholds.py ...` builds `calibration/edge_delta_thresholds.json` + `calibration/defaults.json` (supports sharding via `--shard-manifest/--shard-id`).
- **Ablations**
  - `python -m experiments.ablate.run --config experiments/ablation_matrix.yaml` (or `ablation_matrix_tiny.yaml`) combines precomputed outputs into summary tables; `make rc-ablations` calls this after RC runs.
- **Reporting**
  - `tools/make_summary.py` assembles RC summaries; `tools/summarize_rc_sanity.py` builds completeness + limitations tables; `tools/build_gallery.py` + `build_memo.py` + `build_brief.py` render shareable artefacts; `make gpt-bundle` packages required docs + diff + run log.
- **Monitoring / cache**
  - `.cache/` holds balanced panels and per-window stats for resume; `meta/completeness.py` checks required files before summaries. `run_monitor.py` can tail `metrics.jsonl` if present.
