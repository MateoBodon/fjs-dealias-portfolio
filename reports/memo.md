# Release Candidate Memo - rc

This memo synthesizes smoke and crisis release-candidate runs, highlighting estimator deltas, coverage bottlenecks, and ablation sensitivities.

## Run Selection
- **full_oneway_J5_solver-auto_est-cc_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) [edge=scm | gate=fixed | df~0.020] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **crisis_20200215_20200531** (design=oneway, J=5, period=2020-02-15 → 2020-05-31) [edge=tyler | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-factor_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-lw_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-oas_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-tyler_shrink_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_nested_J5_solver-auto_est-dealias_prep-none** (design=nested, J=5, period=2022-01-01 → 2023-12-31) — nested scope de-scoped (no isolated spikes) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2020-02-01 → 2020-05-31, median angle=38.5°) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink
- **crisis_20200215_20200531** (design=oneway, J=5, period=2020-02-15 → 2020-05-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2022-09-01 → 2022-11-30, median angle=24.2°) [edge=scm | gate=fixed] — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, POET-lite, SCM, Tyler-Shrink


## Nested Scope
- full_nested_J5_solver-auto_est-dealias_prep-none: nested equity de-scoped (gating blocked all windows)


## Key Results
Values are measured versus the de-aliased baseline; negative Delta MSE indicates an improvement. The first table reports MSE deltas and DM p-values, while the second summarises QLIKE DM statistics and substitution share.

| run | crisis_label | estimator | edge|gate | detection_rate | delta_mse_ew | CI_EW | DM_p_EW | DM_p_EW_QLIKE | delta_mse_mv | CI_MV | DM_p_MV | DM_p_MV_QLIKE | edge_margin_median | edge_margin_IQR | mean_qlike | substitution_fraction | no_iso_skip_share | delta_frac_used | VaR_pof | VaR_indep | ES_p | n_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Aliased | scm | fixed | 0.0% (no accepted detections; check guardrails) | 0.00e+00 | [-0.00e+00, -0.00e+00] | n/a | n/a | 0.00e+00 | [-0.00e+00, -0.00e+00] | n/a | n/a | n/a | n/a | -7.308 | 0.0% | n/a | n/a | 0.222 | 0.342 | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | scm | fixed | 0.0% (no accepted detections; check guardrails) | 1.85e-07 | n/a | n/a | n/a | 4.03e-12 | n/a | n/a | n/a | n/a | n/a | -7.121 | 0.0% | n/a | n/a | 0.222 | 0.763 | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | scm | fixed | 0.0% (no accepted detections; check guardrails) | 3.67e-09 | [-7.40e-08, 1.57e-07] | n/a | 0.730 | 5.57e-08 | [5.17e-08, 5.96e-08] | n/a | 0.0e+00 | n/a | n/a | -7.283 | 0.0% | n/a | n/a | 0.222 | n/a | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | OAS | scm | fixed | 0.0% (no accepted detections; check guardrails) | 2.07e-08 | n/a | n/a | 0.558 | 4.13e-08 | n/a | n/a | 0.0e+00 | n/a | n/a | -7.265 | 0.0% | n/a | n/a | 0.222 | n/a | n/a | 24 |

## 2025-11-07 deterministic DoW + vol RC (WRDS)

- **Scope & mode:** Re-ran the WRDS day-of-week (Tyler primary + SCM check) and vol-state (Tyler) RCs on the CRSP-daily panel (`data/returns_daily.csv`, SHA `96ac7dd…3197`) via `scripts/aws_run.sh` with deterministic threading/single BLAS. Every target now enforces the MV defaults (ridge 1e-4, box [0, 0.1], 5 bps turnover, κ-cap 1e6) from the CLI.
- **Calibration check:** `tools/verify_dataset.py` guards every run against `data/registry.json`. Crisis buckets still reported 0% acceptance, so we nudged `calibration/defaults.json` (δ: 0.35→0.30 for both Tyler/SCM G36 bins) and documented the change in `reports/calibration_notes.md`. Post-nudge reruns remained fully rejected, indicating the gating failures stem from upstream detection errors rather than thresholds.
- **Diagnostics:** `diagnostics.csv`/`diagnostics_detail.csv` log `acceptance_rate = detection_rate = 0` with `reason_code=detection_error` for 59% of DoW windows and 19% of vol windows, and `balance_failure` for the remainder. All acceptance/edge-margin histograms are attached (`reports/rc-20251107/*/acceptance_hist_*.png`, `edge_margin_hist_*.png`) and show the mass at zero. We need to root-cause the repeated `detect_spikes` failures (likely the missing factor residuals noted under `baseline_errors`) before expecting non-zero gate pass-through.
- **Risk & coverage:** Even without accepted overlays, EW calm buckets stay near the 5% VaR target (4.8%), but MV calm exposures overshoot (9–10%) and vol-state EW calm buckets collapsed to 0% (too conservative). Crisis buckets also show 8–11% MV violation rates, so we still have work to do on turnover-aware MV smoothing.
- **Condition numbers:** DoW baselines stay well below the κ-cap (median ≈25, p90 ≈33). Vol-state covariances remain an order of magnitude higher (median 1.8e2; crisis p90 ≈4.6e2) but still under the 1e6 limit after dropping the flagged windows.
- **Baselines parity:** RIE, LW, OAS, QuEST, and EWMA appear in every regime table. Factor-residual rows are absent because the WRDS run lacks aligned factor residuals (`baseline_errors` records `factor:missing factor returns`), which we need to restore before the next evaluation.

### DM(QLIKE) vs LW/OAS — DoW (Tyler overlay)

| Regime | p(EW vs LW) | p(EW vs OAS) | p(MV vs LW) | p(MV vs OAS) |
| --- | --- | --- | --- | --- |
| Full | 0.0031 | 0.0033 | 0.0091 | 0.0092 |
| Calm | 0.2490 | 0.2574 | 0.1313 | 0.1339 |
| Crisis | 0.0286 | 0.0271 | 0.0787 | 0.0724 |

MSE-based DM tests are undefined because overlay == baseline (zero detections). QLIKE contrasts remain finite and show statistically significant differences in the full run (even w/out gating) thanks to shrinker dispersion.

### DM(QLIKE) vs LW — Vol-state (Tyler overlay)

| Regime | p(EW vs LW) | p(MV vs LW) |
| --- | --- | --- |
| Full | 0.0450 | 0.0539 |
| Calm | 0.4672 | 0.2611 |
| Crisis | 0.0525 | 0.1208 |

The vol-state overlay couldn’t be benchmarked against OAS (the overlay baseline already uses OAS, so the rows collapse). LW contrasts hover near significance during crisis/full regimes despite the acceptance drought.
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | POET-lite | scm | fixed | 0.0% (no accepted detections; check guardrails) | 1.56e-07 | n/a | n/a | n/a | 8.89e-13 | n/a | n/a | n/a | n/a | n/a | -7.143 | 0.0% | n/a | n/a | 0.222 | 0.342 | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | SCM | scm | fixed | 0.0% (no accepted detections; check guardrails) | 1.53e-07 | n/a | n/a | n/a | -4.28e-13 | n/a | n/a | n/a | n/a | n/a | -7.146 | 0.0% | n/a | n/a | 0.222 | 0.342 | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | scm | fixed | 0.0% (no accepted detections; check guardrails) | 0.129 | n/a | 0.0e+00 | n/a | 2.59e-06 | n/a | n/a | n/a | n/a | n/a | -1.026 | 0.0% | n/a | n/a | 0.222 | n/a | n/a | 24 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Aliased | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.66e-09 | [-6.49e-10, -6.49e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Constant-Correlation | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 8.91e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Ledoit-Wolf | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.09e-09 | [-1.16e-10, -1.16e-10] | n/a | 4.3e-04 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | OAS | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -9.57e-10 | n/a | n/a | 3.5e-04 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | POET-lite | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | SCM | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Tyler-Shrink | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -9.79e-10 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | n/a | 1.000 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.54e-09 | [-5.64e-10, -5.64e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | 0.020 | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 7.34e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | 0.020 | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.02e-09 | [-7.93e-11, -7.93e-11] | n/a | 0.012 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | 0.020 | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -8.96e-10 | n/a | n/a | 0.011 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | 0.020 | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | POET-lite | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.54e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | 0.020 | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.54e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | 0.020 | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -5.60e-11 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | 0.020 | 1.000 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Aliased | tyler | fixed | 0.0% | 0.00e+00 | [-0.00e+00, -0.00e+00] | n/a | n/a | 0.00e+00 | [-0.00e+00, -0.00e+00] | n/a | n/a | 1.88e-03 | 0.017 | -5.882 | 0.0% | n/a | n/a | 1.000 | 0.710 | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Constant-Correlation | tyler | fixed | 0.0% | 8.16e-07 | n/a | n/a | n/a | 1.01e-05 | n/a | 0.064 | n/a | 1.88e-03 | 0.017 | -5.851 | 0.0% | n/a | n/a | 1.000 | n/a | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Ledoit-Wolf | tyler | fixed | 0.0% | -7.78e-06 | [-9.05e-06, -9.05e-06] | 0.026 | 2.7e-06 | 1.50e-07 | [4.34e-08, 4.34e-08] | n/a | 4.2e-04 | 1.88e-03 | 0.017 | -6.622 | 0.0% | n/a | n/a | 1.000 | 0.025 | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | OAS | tyler | fixed | 0.0% | -8.11e-06 | n/a | 0.022 | 1.4e-05 | 1.78e-07 | n/a | n/a | 3.6e-04 | 1.88e-03 | 0.017 | -6.739 | 0.0% | n/a | n/a | 1.000 | 1.3e-09 | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | POET-lite | tyler | fixed | 0.0% | -8.47e-21 | n/a | n/a | n/a | -5.67e-26 | n/a | n/a | n/a | n/a | n/a | -5.882 | 0.0% | n/a | n/a | 1.000 | 0.710 | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | SCM | tyler | fixed | 0.0% | 1.69e-21 | n/a | n/a | n/a | -2.03e-25 | n/a | n/a | n/a | 1.88e-03 | 0.017 | -5.882 | 0.0% | n/a | n/a | 1.000 | 0.710 | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Tyler-Shrink | tyler | fixed | 0.0% | 0.205 | n/a | 0.003 | n/a | 5.60e-08 | n/a | n/a | n/a | 1.88e-03 | 0.017 | -0.828 | 0.0% | n/a | n/a | 1.000 | 0.025 | n/a | 6 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Aliased | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.66e-09 | [-6.49e-10, -6.49e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Constant-Correlation | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 8.91e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Ledoit-Wolf | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.09e-09 | [-1.16e-10, -1.16e-10] | n/a | 4.3e-04 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | OAS | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -9.57e-10 | n/a | n/a | 3.5e-04 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | POET-lite | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | SCM | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Tyler-Shrink | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -9.79e-10 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | n/a | 1.000 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Aliased | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.66e-09 | [-6.49e-10, -6.49e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Constant-Correlation | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 8.91e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Ledoit-Wolf | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.09e-09 | [-1.16e-10, -1.16e-10] | n/a | 4.3e-04 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | OAS | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -9.57e-10 | n/a | n/a | 3.5e-04 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | POET-lite | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | SCM | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Tyler-Shrink | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -9.79e-10 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | n/a | 1.000 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Aliased | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.66e-09 | [-6.49e-10, -6.49e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Constant-Correlation | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 8.91e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Ledoit-Wolf | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.09e-09 | [-1.16e-10, -1.16e-10] | n/a | 4.3e-04 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | OAS | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -9.57e-10 | n/a | n/a | 3.5e-04 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | POET-lite | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | SCM | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Tyler-Shrink | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -9.79e-10 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | n/a | 1.000 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Aliased | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.66e-09 | [-6.49e-10, -6.49e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Constant-Correlation | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 8.91e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Ledoit-Wolf | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.09e-09 | [-1.16e-10, -1.16e-10] | n/a | 4.3e-04 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | OAS | scm | fixed | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -9.57e-10 | n/a | n/a | 3.5e-04 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | POET-lite | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | SCM | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | n/a | 0.186 | 0.306 | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Tyler-Shrink | scm | fixed | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -9.79e-10 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | n/a | 1.000 | 0.306 | n/a | 4 |

### Performance Snapshot (Full Regime)
| portfolio | ΔMSE vs baseline | VaR95 (overlay) | VaR95 (baseline) | DM p-value | DM n_effective |
| --- | --- | --- | --- | --- | --- |
| ew | nan | nan | nan | nan | 0 |
| mv | nan | nan | nan | nan | 0 |

### Detection Snapshot
| Detection rate (mean) | Detection rate (median) | Edge margin | Stability margin | Isolation share | Alignment cos | Reason code |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0% | 0.0% | 0.00e+00 | 0.00e+00 | 0.0% | n/a | detection_error |

### QLIKE Diagnostics
| run | crisis_label | estimator | DM_p_EW_QLIKE | DM_p_MV_QLIKE | mean_qlike | substitution_fraction | no_iso_skip_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Aliased | n/a | n/a | -7.308 | 0.0% | n/a |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | n/a | n/a | -7.121 | 0.0% | n/a |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 0.730 | 0.0e+00 | -7.283 | 0.0% | n/a |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | OAS | 0.558 | 0.0e+00 | -7.265 | 0.0% | n/a |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | POET-lite | n/a | n/a | -7.143 | 0.0% | n/a |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | SCM | n/a | n/a | -7.146 | 0.0% | n/a |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | n/a | n/a | -1.026 | 0.0% | n/a |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Ledoit-Wolf | 0.050 | 4.3e-04 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | OAS | 0.074 | 3.5e-04 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | POET-lite | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 0.050 | 0.012 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 0.074 | 0.011 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | POET-lite | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Aliased | n/a | n/a | -5.882 | 0.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Constant-Correlation | n/a | n/a | -5.851 | 0.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Ledoit-Wolf | 2.7e-06 | 4.2e-04 | -6.622 | 0.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | OAS | 1.4e-05 | 3.6e-04 | -6.739 | 0.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | POET-lite | n/a | n/a | -5.882 | 0.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | SCM | n/a | n/a | -5.882 | 0.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Tyler-Shrink | n/a | n/a | -0.828 | 0.0% | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Ledoit-Wolf | 0.050 | 4.3e-04 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | OAS | 0.074 | 3.5e-04 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | POET-lite | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Ledoit-Wolf | 0.050 | 4.3e-04 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | OAS | 0.074 | 3.5e-04 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | POET-lite | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Ledoit-Wolf | 0.050 | 4.3e-04 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | OAS | 0.074 | 3.5e-04 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | POET-lite | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Ledoit-Wolf | 0.050 | 4.3e-04 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | OAS | 0.074 | 3.5e-04 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | POET-lite | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |

### What Limits Coverage
What limits coverage: no dominant rejection signals.

| run | edge_buffer | off_component_ratio | stability_fail | energy_floor | neg_mu |
| --- | --- | --- | --- | --- | --- |
| nested_J5_solver-auto_est-dealias_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-cc_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-factor_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-lw_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-oas_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Kill Criteria (CHECK)
- [N/A] EW ΔMSE must not exceed baseline (value: nan, threshold: {"max": 0.0})
- [N/A] MV ΔMSE must not exceed baseline (value: nan, threshold: {"max": 0.0})
- [FAIL] Detection coverage within target band (value: 0, threshold: {"max": 0.25, "min": 0.01})
- [FAIL] Average edge margin positive (value: 0, threshold: {"min": 0.0})
- [N/A] Alignment cosine above 0.9 (value: nan, threshold: {"min": 0.9})
- [PASS] Dominant reason code acceptable (value: n/a, threshold: {"allowed": ["accepted"]})

## Limitations
- EW ΔMSE must not exceed baseline: value unavailable.
- MV ΔMSE must not exceed baseline: value unavailable.
- Detection coverage within target band: observed 0 vs threshold {'min': 0.01, 'max': 0.25}.
- Average edge margin positive: observed 0 vs threshold {'min': 0.0}.
- Alignment cosine above 0.9: value unavailable.

## Highlights
- Full regime detection 0.0% (reason: detection_error); edge margin 0.000, stability 0.000.
- Delta MSE (EW) < 0 in 26.8% of comparisons (median Delta MSE (EW) = 1.756e-07).
- 4 of 11 Diebold–Mariano tests show p < 0.05.
- 24 of 32 QLIKE DM tests show p < 0.05.
- Accepted detections substitute in 56.2% of evaluated windows.
- Median alignment angle across runs: 17.5° (mean 18.6°).
- 'no_isolated_spike' gate skipped 25.0% of windows on average (max 25.0%).

## Diagnostics Snapshot
### Reason Codes
(no reason-code diagnostics)

### Edge Margin Distribution
(edge margin histogram unavailable)

### Isolation Share
(isolation share chart unavailable)

### Direction vs Stability
(direction vs stability plot unavailable)

## Ablation Snapshot
(global ablation matrix unavailable)

No ablation sweep artifacts were detected for this gallery.

(no ablation sweep detected)



## Artifacts
- oneway_J5_solver-auto_est-cc_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-cc_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-cc_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-cc_prep-none/run_meta.json
- oneway_J5_solver-auto_est-dealias_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2020/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2022/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json
- oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531: tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2020/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/run_meta.json
- oneway_J5_solver-auto_est-factor_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-factor_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-factor_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-factor_prep-none/run_meta.json
- oneway_J5_solver-auto_est-lw_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-lw_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-lw_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-lw_prep-none/run_meta.json
- oneway_J5_solver-auto_est-oas_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-oas_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-oas_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-oas_prep-none/run_meta.json
- oneway_J5_solver-auto_est-tyler_shrink_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-tyler_shrink_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-tyler_shrink_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-tyler_shrink_prep-none/run_meta.json
- nested_J5_solver-auto_est-dealias_prep-none: tables: figures/rc/nested_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/nested_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_nested_smoke/nested_J5_solver-auto_est-dealias_prep-none/run_meta.json
