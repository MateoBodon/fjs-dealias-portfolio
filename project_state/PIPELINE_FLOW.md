---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Pipeline Flow

## Daily evaluation (RC / RC-lite / RC-lite-sanity)
- Entry point: `experiments/eval/run.py` (CLI via `python -m experiments.eval.run` or `PYTHONPATH=src:. python experiments/eval/run.py`).
- Make targets: `make rc`, `make rc-lite`, `make rc-lite-sanity`, `make rc-dow`, `make rc-vol`, `make rc-week`, `make rc-dowxvol`.
- Typical flow:
  1) Verify datasets (`tools/verify_dataset.py` invoked by Makefile).
  2) Run eval (`experiments/eval/run.py`) → outputs under `reports/<run>/` with `run.json` + `resolved_config.json`.
  3) Summarize (`tools/make_summary.py`) and RC-lite sanity (`tools/summarize_rc_sanity.py`).
- Validity: `run.json` windows block sets `cap_active`/`cap_sources`; summaries exclude capped runs and require `comparison_valid_*` + `n_effective_*`.

## Weekly equity panel
- Entry point: `experiments/equity_panel/run.py` with YAML configs in `experiments/equity_panel/`.
- Make targets: `make run:equity_smoke`, `make run:equity_nested_smoke_tiny`, `make run-equity`, `make rc-lite` (weekly legs).
- Outputs: `experiments/equity_panel/outputs_*` with `config_resolved.yaml`, `detection_summary.csv`, `gating_diagnostics.csv` (if enabled), plots.

## Synthetic calibration
- Entry points: `experiments/synthetic/null.py`, `power.py`, `power_null.py`, `nested_killtest.py`, `calibrate_thresholds.py`.
- Make targets: `make calibrate-thresholds`, `make sweep:acceptance`.
- Outputs: `reports/synthetic/*`, `reports/figures/*`, calibration JSONs in `calibration/`.

## Reporting & memos
- Gallery/memo/brief: `tools/build_gallery.py`, `tools/build_memo.py`, `tools/build_brief.py` (often via `make gallery` / `make memo`).
- Summary tables: `tools/make_summary.py`, `tools/summarize_weekly_diagnostics.py`.

## Packaging / audit
- `make gpt-bundle TICKET=<id> RUN_NAME=<run>` builds a shareable zip of docs + run log (see `docs/DOCS_AND_LOGGING_SYSTEM.md`).
