---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Dataflow

- **Inputs (registries enforced)**
  - Returns: `data/returns_daily.csv` (sha256 `96ac7dd3…3197`, 300 cols, 2010-01-05→2024-12-31) registered in `data/registry.json`.
  - Factors: `data/factors/ff5mom_daily.csv` (sha256 `469d44ad…908ca`, MKT/SMB/HML/RMW/CMA/MOM/RF, 2005-01-03→2025-08-29) in `data/factors/registry.json`.
  - Verification: `python tools/verify_dataset.py data/returns_daily.csv --registry data/registry.json` (same for factors) is wired into `make rc*` targets.
- **Preprocessing**
  - Balancing: `eval.balance.build_balanced_window` aligns panels; partial-week handling via CLI (`--drop-partial-weeks` default, `--impute-partial-weeks` optional).
  - Prewhitening: `experiments/prewhiten.py` and `experiments/eval/run.py` apply factor regression; factor fallbacks defined in `experiments/prewhiten.py` and `experiments/eval/config.py`.
  - Caching: `.cache/rc-lite` and `.cache/mp_edges` store balanced panels and MP edge grids; cache keys include design/estimator/preprocessing flags.
- **Calibration artefacts**
  - `calibration/defaults.json` (energy-floor, thresholds) and `calibration/edge_delta_thresholds.json` produced by `make calibrate-thresholds`.
  - Synthetic outputs: `reports/synthetic/{null_harness,power_harness}/`, `reports/figures/roc_null.png`, `reports/figures/roc_power.png`.
- **Experiment outputs**
  - Daily eval: `reports/rc-YYYYMMDD*/` (rc/rc-lite/rc-lite-sanity) containing `metrics_summary.csv`, `diagnostics*.csv`, DM tables, plots, `resolved_config.json`, `run.json`.
  - Weekly panels: `experiments/equity_panel/outputs_*` per design/estimator, each with `config_resolved.yaml`, `detection_summary.csv`, spectra/VaR plots, gating diagnostics; avoid overwriting historical outputs.
  - Crisis/sweep artefacts: `reports/rc-20251121/`, `reports/rc-20251113/`, etc. Only sample or read metadata; do not delete.
- **Reporting**
  - Summaries in `reports/rc-YYYYMMDD/summary*.{json,csv,md}`; galleries under `figures/rc/YYYYMMDD/`; memos/briefs in `reports/` with timestamped copies.
- **Logs & manifests**
  - Each run writes `resolved_config.json`/`run.json` (eval) or `config_resolved.yaml`/`run_meta.json` (equity_panel). Completeness checks in `meta/completeness.py` guard summaries.
- **Exclusions**
  - Do not modify `data/wrds/` mounts or raw WRDS exports. Large dirs (`reports/`, `data/`, `experiments/equity_panel/outputs_*`) were not parsed deeply for this rebuild; rely on manifests.
