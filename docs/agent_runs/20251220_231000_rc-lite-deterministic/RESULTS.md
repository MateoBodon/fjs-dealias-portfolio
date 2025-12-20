# Results
- RC-lite sanity (deterministic): `reports/rc-20251220-sanity-20251220_233700/`
  - cap_active=false; no cap_sources; high coverage (n_effective_mse=33 for EW/MV overlay rows).
  - Key validity: `comparison_valid_mse=1`, `comparison_valid_qlike=1` for overlay (EW & MV), with `n_effective_mse=33`, `n_effective_qlike=33` (file: dow-tyler/full/metrics.csv, rows estimator=overlay).
  - Delta comparisons in summary marked invalid because ΔMSE > 0 (expected given overlay underperforms); validity for metrics themselves is true.
- Weekly gating diagnostics (generated with --gating-diagnostics):
  - Dow weekly: `experiments/equity_panel/outputs_rc-lite-20251220_20251220_233700/dow-weekly/.../gating_diagnostics.csv` — 4 windows, skip_reason_primary all `calibration_missing_p_T`, guard_unknown sum=0, exception_type/stage empty (no failures).
  - Nested weekly: `experiments/equity_panel/outputs_rc-lite-20251220_20251220_233700/nested/.../gating_diagnostics.csv` — 10 windows, skip_reason_primary all `calibration_missing_p_T` with edge/p/t detail, guard_unknown sum=0.
- Summary artifacts: `summary/summary_perf.csv`, `summary/summary_detection.csv`, `summary/limitations.md`, `summary/completeness.json` (complete, uncapped). Gallery/memo refreshed.
