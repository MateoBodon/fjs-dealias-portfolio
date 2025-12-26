- Implemented component-aware inject_mode (total/between/within) with group-aware series construction and added inject_mode + series/group summaries to run.json.
- Outputs now include inject_mode in curve.csv, windows_detail.csv, gating_reasons.csv.
- Added unit tests for inject_mode semantics and updated CSV schema tests.

Run summary (between-mode smoke, week design):
- Run dir: reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/
- Artifacts copied: docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/artifacts/
- Required outputs present: run.json, resolved_config.json, curve.csv, windows_detail.csv, gating_reasons.csv.

Acceptance check:
- curve.csv includes baseline μ=0.0 and injected μ={12,24} rows.
- μ=24.0 detection_rate=1.00 (2/2), acceptance_rate=1.00 (2/2).
- windows_detail.csv shows pre_gate_raw_outliers_found>0 for 100% of μ=24 injected windows (2/2).

Detection/acceptance by μ (curve.csv):
- μ=0.0: detect=0.00, accept=0.00 (n_windows=20)
- μ=12.0: detect=1.00, accept=1.00 (n_windows=2)
- μ=24.0: detect=1.00, accept=1.00 (n_windows=2)

Top gating reasons (gating_reasons.csv):
- μ=0.0 pre-gate: tvec_off_component=8000, tvec_no_real_root=1671, tvec_no_admissible_root=1079, off_component_ratio=250.
- μ=24.0 pre-gate: tvec_off_component=826, tvec_no_real_root=190, tvec_no_admissible_root=98; post-gate accepted=2.

Bundle:
- docs/gpt_bundles/20251226_101751_ticket-25_20251226_095630_ticket-25_week-between-smoke.zip
