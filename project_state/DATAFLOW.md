---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Dataflow

- **Inputs (registries enforced)**
  - Returns: `data/returns_daily.csv` (sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`, rows 892529, columns 300, 2010-01-05→2024-12-31) registered in `data/registry.json`.
  - Factors: `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`, columns ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF'], 2005-01-03→2025-08-29) registered in `data/factors/registry.json`.
  - Verification: `python tools/verify_dataset.py data/returns_daily.csv --registry data/registry.json` (and factors) wired into Make targets.
- **Preprocessing**
  - Balancing: `eval.balance.build_balanced_window` (daily eval) and `src/data/panels.py` (weekly panel) align/shape windows; partial-week policy via `--drop-partial-weeks`/`--impute-partial-weeks`.
  - Prewhitening: `experiments/prewhiten.py` and `experiments/eval/run.py` apply factor regression; factor fallbacks in `experiments/prewhiten.py`.
  - Caching: `.cache/` directories store MP edges, balanced panels, and per-window stats when `--resume`/`--precompute-panel` is enabled.
- **Calibration artifacts**
  - `calibration/defaults.json` and `calibration/edge_delta_thresholds.json` from synthetic sweeps (`make calibrate-thresholds`, `make sweep:acceptance`).
  - Nested calibration: `calibration/nested_edge_delta_thresholds.json` from `experiments/synthetic/nested_killtest.py`.
- **Experiment outputs**
  - Daily eval: `reports/rc-YYYYMMDD*/` with `metrics*.csv`, `diagnostics*.csv`, DM tables, `resolved_config.json`, `run.json`.
  - Weekly panels: `experiments/equity_panel/outputs_*` with `config_resolved.yaml`, `panel_manifest.json`, `detection_summary.csv`, plots and (optional) gating diagnostics.
  - Synthetic outputs: `reports/synthetic/` + `reports/figures/`.
- **Reporting**
  - Summary tables under `reports/*/summary/` plus memo/gallery outputs in `reports/` and `figures/rc/`.
- **Exclusions**
  - Raw WRDS exports are not committed; large dirs (`reports/`, `data/`, `experiments/equity_panel/outputs_*`) were not parsed deeply for this rebuild; rely on manifests/registries.
