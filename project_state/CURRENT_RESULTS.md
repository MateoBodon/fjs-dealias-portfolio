---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Current Results (latest validated drops)

- **2025-12-21 — Daily DoW paper-v1 attempt (ticket-05)**
  - Run: `reports/rc-ticket-05-20251221_221902/dow-paper-v1/` with `cap_active=true` (`cap_sources=['window_coverage']`).
  - Summary tables are empty (`summary_perf.csv`/`summary_detection.csv`), so this run is **not headline-valid**.
  - Artifacts: `reports/rc-ticket-05-20251221_221902/summary/overlay_forensics.csv` (6997 rows) and `summary/limitations.md` list the cap.

- **2025-12-20 — RC-lite sanity (deterministic)**
  - Run: `reports/rc-20251220-sanity-20251220_233700/` (uncapped; `cap_active=false`).
  - Overlay metrics valid in per-regime CSVs (`comparison_valid_mse=1`, `comparison_valid_qlike=1`, `n_effective=33` for EW/MV overlay rows; see `dow-tyler/full/metrics.csv`).
  - Summary outputs present: `summary/summary_perf.csv`, `summary/summary_detection.csv`, `summary/completeness.json`.

- **2025-12-20 — Weekly gating attribution smoke (ticket-09)**
  - Run: `experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`.
  - Detection rate 0.75 (3/4); `guard_unknown` column present with sum 0; `guard_other` absent; skip reason `no_isolated_spike` for the lone rejection.

- **2025-12-20 — Nested calibration audit (ticket-14)**
  - `calibration/nested_edge_delta_thresholds.json` embeds audit metadata; tiny nested smoke (`experiments/equity_panel/outputs_nested_smoke_tiny/`) skips windows with `calibration_missing_p_T` (p≈188, T=70/80 outside calibrated grid).

Older RC-lite (2025-11-21) and prewhitening/vol acceptance studies remain under `reports/rc-20251121/` and `reports/rc-20251113/`; see PROGRESS.md for provenance.
