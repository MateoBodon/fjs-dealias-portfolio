---
generated: 2026-02-16T02:32:01Z
git_sha: 1371b3c2e7197c3629cc20e4e67c1f435f3ca13a
git_branch: codex/ticket-27-repo-hygiene-cleanup
commands:
  - manual documentation recenter for ticket-31
  - artifact verification from reports/* and calibration/* files
---
# Current Results (artifact-verified snapshot)

## 2025-12-26 - Injection sensitivity remains flat-zero (ticket-24)

- Run artifact: `reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv`.
- Verified curve rows:
  - `mu=0.0`: detection_rate=0.0, acceptance_rate=0.0, n_windows=186
  - `mu=3.0`: detection_rate=0.0, acceptance_rate=0.0, n_windows=74
  - `mu=6.0`: detection_rate=0.0, acceptance_rate=0.0, n_windows=74
  - `mu=12.0`: detection_rate=0.0, acceptance_rate=0.0, n_windows=74
  - `mu=24.0`: detection_rate=0.0, acceptance_rate=0.0, n_windows=74
- Dominant pre-gate reasons (verified in `reports/inject_spike/20251226_ticket24_week_full_fix/gating_reasons.csv`): `tvec_off_component`, `tvec_no_real_root`, `tvec_no_admissible_root`.

## 2025-12-23 - Nested calibration coverage refresh (ticket-17, git b2221e8)

- Run artifact: `reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/run.json`.
- Verified selection metrics:
  - `null_trials=440`
  - `null_rate=0.0`
  - `null_ci_high=0.01716151619513562`
  - `power_moderate=1.0`
  - `power_strong=1.0`
- Calibration artifact: `calibration/nested_edge_delta_thresholds.json` includes p=188 and p=200 entries for T in {60,70,80}.
- Tiny nested smoke artifact: `experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/weekly_diagnostics.md`.
  - Verified skip reasons across 3 windows: `instability_in_a_neighborhood` (2), `no_isolated_spike` (1).
  - `calibration_missing_p_T` is not present in this smoke output.

## 2025-12-22 - Daily DoW paper-v1 uncapped run (ticket-07, git 2cb5bfd)

- Detection artifact: `reports/rc-ticket-07-20251222_183800/summary/summary_detection.csv`.
  - `cap_active=False`, `window_coverage=1.0`
  - `windows=1774`, `detection_windows=1751`
  - `detection_rate_mean=0.0416229200503975` (4.16%)
  - Note: `detection_rate_mean` and `detection_windows/windows` are distinct fields in this output; do not treat them as the same ratio.
- Performance artifact: `reports/rc-ticket-07-20251222_183800/summary/summary_perf.csv`.
  - EW full-regime: `delta_mse_vs_baseline=2.635418515787517e-11`, `delta_qlike_vs_baseline=-0.0671866909475027`, `n_effective_mse=1749.0`
  - MV full-regime: `delta_mse_vs_baseline=-6.654496181059978e-13`, `delta_qlike_vs_baseline=-0.0357629174555866`, `n_effective_mse=1749.0`

## 2025-12-22 - Daily DoW paper-v1 prior drop (ticket-06, git 8a5579b)

- Detection artifact: `reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv`.
  - `cap_active=False`, `window_coverage=1.0`
  - `windows=1774`, `detection_windows=1751`
  - `detection_rate_mean=0.0416229200503975` (4.16%)
- Performance artifact: `reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv`.
  - EW full-regime: `delta_mse_vs_baseline=2.635418515787517e-11`
  - MV full-regime: `delta_mse_vs_baseline=-6.654496181059978e-13`

## 2025-11-21 - Deterministic RC-lite reference (non-headline due cap)

- Artifact: `reports/rc-20251121/metrics_summary.json`.
- Verified run-level values:
  - `dow-tyler`: `detection_rate=0.0431992716914086`, `acceptance_rate=0.0431992716914086`, `delta_mse_ew=1.749123263776732e-13`, `delta_mse_mv=-2.5377866153688694e-14`
  - `vol-tyler`: `detection_rate=0.043346108100964724`, `acceptance_rate=0.043346108100964724`, `delta_mse_ew=-1.054439874524572e-13`, `delta_mse_mv=-8.637974669528799e-14`
- Citing rule: this RC-lite drop was window-capped for throughput and is useful for diagnostics, not primary headline claims.
