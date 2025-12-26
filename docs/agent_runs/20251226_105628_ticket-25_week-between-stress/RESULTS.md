- Proposed stress-test default: use all available windows (max-windows=48), between-mode, μ grid 0,6,12,18,24,30,36, inject_frac fixed at 0.2 for more injected windows per μ.
- Ran between-mode stress test on fixture panel; outputs and artifacts captured.

Run summary:
- Run dir: reports/inject_spike/20251226_105628_ticket-25_week-between-stress/
- Artifacts copied: docs/agent_runs/20251226_105628_ticket-25_week-between-stress/artifacts/
- Required outputs present: run.json, resolved_config.json, curve.csv, windows_detail.csv, gating_reasons.csv.

Acceptance check:
- curve.csv includes μ=0.0 and μ={6,12,18,24,30,36} rows.
- μ=36.0 detection_rate=1.00 (10/10), acceptance_rate=1.00 (10/10).
- windows_detail.csv shows pre_gate_raw_outliers_found>0 for 100% of μ=36 injected windows (10/10).

Detection/acceptance by μ (curve.csv):
- μ=0.0: detect=0.00, accept=0.00 (n_windows=48)
- μ=6.0: detect=1.00, accept=1.00 (n_windows=10)
- μ=12.0: detect=1.00, accept=1.00 (n_windows=10)
- μ=18.0: detect=1.00, accept=1.00 (n_windows=10)
- μ=24.0: detect=1.00, accept=1.00 (n_windows=10)
- μ=30.0: detect=1.00, accept=1.00 (n_windows=10)
- μ=36.0: detect=1.00, accept=1.00 (n_windows=10)

Top gating reasons (gating_reasons.csv):
- μ=0.0 pre-gate: tvec_off_component=19200, tvec_no_real_root=4062, tvec_no_admissible_root=2588, off_component_ratio=550.
- μ=36.0 pre-gate: tvec_off_component=4130, tvec_no_real_root=790, tvec_no_admissible_root=506, off_component_ratio=144; post-gate accepted=10.

Bundle:
- docs/gpt_bundles/20251226_110750_ticket-25_20251226_105628_ticket-25_week-between-stress.zip
