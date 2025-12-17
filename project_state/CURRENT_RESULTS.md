# Current Results (as of 2025-12-17)

- **rc-lite-sanity (stamp 20251209_001356)** — `reports/rc-20251208-sanity-20251209_001356/`
  - Daily DoW (tyler, rie, 60×10, assets_top=50): detection_rate_mean ≈ 5.36%, acceptance≈detection, percent_changed=100%, edge_margin_mean ≈ 0.00365. ΔMSE vs baseline positive (EW 6.93e-11, MV 1.76e-11), so kill criteria fail (`kill_criteria.json`). Alignment cosine ≈1.0; isolation=1.0. Prewhiten R² mean ≈0.317 (FF5+MOM).
  - Daily Vol-state (tyler, oas): detection_rate ≈ 5.22%, acceptance≈5.56%, percent_changed≈93.9%; edge_margin_mean ≈0.00376. ΔMSE not summarised in summary tables yet (only DoW aggregated); metrics.csv shows small deltas, overlay still alters most windows.
  - Regime splits: calm detection ~4.9–5.5%, crisis detection ~5.5%; DM stats empty (n_effective 32 full).
  - Summary tables/limitations present; memo/brief not regenerated for this batch.
- **Weekly rc-lite-sanity smoke (same stamp)** — `experiments/equity_panel/outputs_rc-lite-20251208_20251209_001356/`
  - DoW weekly (2023Q1, J=5, window=6, horizon=1, tyler): rolling_windows_evaluated=4; detection_windows=0; substitution_fraction=0.
  - Nested weekly (2022–2023H1, window=52, horizon=1): rolling_windows_evaluated=10; detection_windows=0; substitution_fraction=0. Highlights persistent nested/weekly detection drought at current guardrails.
- **Older full RC-lite (capped 200 windows)** — `reports/rc-20251121/` remains last completed full RC-lite run; detection≈4.32–4.33% (DoW/Vol), percent_changed≈100%, ΔMSE near 0 (see README/REPORT). Treat as stale relative to rc-lite-sanity.
- **Calibration defaults** — `calibration_defaults.json` generated 2025-11-21 (SCM energy_floor≈0.108; delta=0.5, delta_frac=0.02, eps=0.02, stability_eta_deg=0.4); `calibration/edge_delta_thresholds.json` unchanged since same date.
- **Gaps**
  - `reports/rc-20251208/` contains only resolved_config/prewhiten files (no metrics); likely incomplete run.
  - rc-lite-sanity summary aggregates DoW only (vol run not folded into summary/kill criteria).
  - No recent crisis or full-length nested RC after November; crisis performance and nested acceptance remain unknown under current thresholds.
