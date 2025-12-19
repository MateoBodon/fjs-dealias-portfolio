# Current Results (as of 2025-12-19)

- **Weekly gating diagnostics (ticket-07, RUN_NAME=20251219_173231_ticket-07_weekly-drought-diagnostics)** — diagnostics artifact + summaries
  - Real-data DoW smoke (config.smoke.yaml with `--gating-diagnostics`, output `experiments/equity_panel/outputs_smoke_ticket07_20251219_173231/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`): detection_rate = 0.75 (3/4 windows), single skip_reason = `no_isolated_spike`; guardrail counts dominated by `guard_other` (=1148). Summary: `weekly_diagnostics.md` in the run directory.
  - Synthetic micro smoke (generated returns via `/tmp/ticket07_synth_returns.csv`, config `docs/agent_runs/20251219_173231_ticket-07_weekly-drought-diagnostics/config.synthetic.yaml`, output `experiments/equity_panel/outputs_ticket07_synth_20251219_173231/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`): detection_rate = 0, skip_reason = `diagnostic_failure` for all 6 windows; summary co-located `weekly_diagnostics.md`.

- **rc-lite-sanity (stamp 20251209_001356)** — `reports/rc-20251208-sanity-20251209_001356/`
  - Daily DoW (tyler, rie, 60×10, assets_top=50): detection_rate_mean ≈ 5.36%, acceptance≈detection, percent_changed=100%, edge_margin_mean ≈ 0.00365. ΔMSE vs baseline remains positive (EW ≈1.24e-10, MV ≈4.52e-11); summary_sanity overlay_effect = *harmful*. Alignment cosine ≈1.0; isolation=1.0. Prewhiten R² mean ≈0.317 (FF5+MOM).
  - Daily Vol-state (tyler, oas): detection_rate ≈ 5.22%, acceptance≈5.56%, percent_changed≈93.9%; edge_margin_mean ≈0.00376. ΔMSE now tabulated (EW ≈3.67e-11, MV ≈1.24e-13), also **harmful** with overlay_effect = harmful.
  - Regime splits: calm detection ~4.9–5.5%, crisis detection ~5.5%; DM stats empty (n_effective 32 full).
  - Summary_sanity/regime.csv regenerated with daily DoW + vol and weekly dow/nested; memo/brief still stale for this batch.
- **Weekly rc-lite-sanity smoke (same stamp)** — `experiments/equity_panel/outputs_rc-lite-20251208_20251209_001356/`
  - DoW weekly (2023Q1, J=5, window=6, horizon=1, tyler): rolling_windows_evaluated=4; detection_windows=0; substitution_fraction=0.
  - Nested weekly (2022–2023H1, window=52, horizon=1): rolling_windows_evaluated=10; detection_windows=0; substitution_fraction=0. Highlights persistent nested/weekly detection drought at current guardrails.
- **Older full RC-lite (capped 200 windows)** — `reports/rc-20251121/` remains last completed full RC-lite run; detection≈4.32–4.33% (DoW/Vol), percent_changed≈100%, ΔMSE near 0 (see README/REPORT). Treat as stale relative to rc-lite-sanity.
- **Calibration defaults** — `calibration_defaults.json` generated 2025-11-21 (SCM energy_floor≈0.108; delta=0.5, delta_frac=0.02, eps=0.02, stability_eta_deg=0.4). `calibration/edge_delta_thresholds.json` refreshed 2025-12-17 with direct entries for tyler + huber covering p∈{188,200}, T∈{60,70,80} (plus backfilled 64/96) to unblock nested calibrated gating.
- **Nested synthetic kill-test (20251217)** — `reports/synthetic_nested_killtest/summary.md`: under the current nested settings (p≈200, weeks 6–8, reps=5, delta=0.35, delta_frac_min=0.05) the null scenario still accepts 100% of windows (FPR≈1.0), i.e., overlay is unsafe; power indistinguishable because acceptance is unconditional.
- **Gaps**
  - `reports/rc-20251208/` contains only resolved_config/prewhiten files (no metrics); likely incomplete run.
  - rc-lite-sanity daily overlay harmful on both DoW and vol slices; kill criteria failing (ΔMSE>0) despite low detection coverage.
  - No recent crisis or full-length nested RC after November; crisis performance and nested acceptance remain unknown under current thresholds.

#### Daily rc-lite-sanity (2023H1, 50 assets, DoW + vol-state)

Status as of 2025-12-17 (run: `reports/rc-20251208-sanity-20251209_001356/`):

- **DoW daily (Tyler edge, FF5+MOM prewhitened)**  
  - detection_rate ≈ 5.36% of windows  
  - percent_changed ≈ 100% (overlay replaces the covariance whenever it triggers)  
  - ΔMSE (EW, MV) > 0 vs Ledoit–Wolf on this slice  
  - Under our current kill criteria, this counts as **harmful overlay**: whenever the overlay fires, it tends to worsen OOS variance.

- **Vol-state daily (Tyler edge, FF5+MOM prewhitened)**  
  - detection_rate ≈ 5.22% of windows  
  - percent_changed ≈ 93.9%  
  - ΔMSE (EW ≈ 3.7e-11, MV ≈ 1.2e-13) > 0 vs Ledoit–Wolf  
  - Overlay flagged **harmful** in summary_sanity (harm fails kill criteria).

- **Weekly DoW + nested smokes**  
  - DoW (config.smoke.yaml with diagnostics, outputs_smoke_ticket07_20251219_173231): detection_windows=3/4, skip_reason `no_isolated_spike` on the lone rejection; guardrail tally shows `guard_other`=1148 in gating_diagnostics.csv.  
  - Nested acceptance in rc-lite-sanity (20251209_001356) remains 0/10, still dominated by `no_isolated_spike` and calibration gaps for p≈188, T≈60–80. Diagnostics added but behaviour unchanged; nested remains **non-functional** in WRDS runs.
