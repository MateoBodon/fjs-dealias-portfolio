# Release Candidate Memo - smoke

This memo synthesizes smoke and crisis release-candidate runs, highlighting estimator deltas, coverage bottlenecks, and ablation sensitivities.

## Run Selection
- **full_oneway_J5_solver-auto_est-cc_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31, median angle=17.5° [no_isolated 25%]) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **crisis_20200215_20200531** (design=oneway, J=5, period=2020-02-15 → 2020-05-31, median angle=31.0°) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-factor_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-lw_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-oas_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-tyler_shrink_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink

## Key Results
Values are measured versus the de-aliased baseline; negative Delta MSE indicates an improvement. The first table reports MSE deltas and DM p-values, while the second summarises QLIKE DM statistics and substitution share.

| run | crisis_label | estimator | detection_rate | delta_mse_ew | CI_EW | DM_p_EW | DM_p_EW_QLIKE | delta_mse_mv | CI_MV | DM_p_MV | DM_p_MV_QLIKE | edge_margin_median | edge_margin_IQR | mean_qlike | substitution_fraction | no_iso_skip_share | n_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | n/a | -8.08e-08 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | n/a | -1.72e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | [2.32e-07, 2.32e-07] | n/a | n/a | -1.66e-09 | [-6.49e-10, -6.49e-10] | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 75.0% ⚠ gate-no_iso 25% | 1.67e-07 | n/a | n/a | n/a | 8.91e-08 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.622 | 75.0% | 25.0% | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 75.0% ⚠ gate-no_iso 25% | -1.17e-07 | [-1.07e-07, -1.07e-07] | n/a | 0.050 | -1.09e-09 | [-1.16e-10, -1.16e-10] | n/a | 4.3e-04 | 3.74e-03 | 2.66e-03 | -8.493 | 75.0% | 25.0% | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 75.0% ⚠ gate-no_iso 25% | -1.28e-07 | n/a | n/a | 0.074 | -9.57e-10 | n/a | n/a | 3.5e-04 | 3.74e-03 | 2.66e-03 | -8.743 | 75.0% | 25.0% | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | 75.0% ⚠ gate-no_iso 25% | 2.60e-07 | n/a | n/a | n/a | -1.66e-09 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -7.583 | 75.0% | 25.0% | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 75.0% ⚠ gate-no_iso 25% | 0.086 | n/a | 0.074 | n/a | -9.79e-10 | n/a | n/a | n/a | 3.74e-03 | 2.66e-03 | -1.351 | 75.0% | 25.0% | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Aliased | 100.0% | 2.22e-06 | [2.47e-06, 2.47e-06] | n/a | n/a | -2.68e-07 | [-2.95e-07, -2.95e-07] | n/a | n/a | 4.24e-03 | 2.24e-03 | -5.882 | 100.0% | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Constant-Correlation | 100.0% | 3.04e-06 | n/a | 0.040 | n/a | 9.85e-06 | n/a | 0.068 | n/a | 4.24e-03 | 2.24e-03 | -5.851 | 100.0% | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Ledoit-Wolf | 100.0% | -5.56e-06 | [-6.42e-06, -6.42e-06] | 0.024 | 7.6e-06 | -1.18e-07 | [-1.16e-07, -1.16e-07] | n/a | 2.5e-07 | 4.24e-03 | 2.24e-03 | -6.622 | 100.0% | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | OAS | 100.0% | -5.89e-06 | n/a | 0.020 | 6.7e-05 | -9.02e-08 | n/a | n/a | 1.8e-07 | 4.24e-03 | 2.24e-03 | -6.739 | 100.0% | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | SCM | 100.0% | 2.22e-06 | n/a | n/a | n/a | -2.68e-07 | n/a | n/a | n/a | 4.24e-03 | 2.24e-03 | -5.882 | 100.0% | n/a | 6 |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Tyler-Shrink | 100.0% | 0.205 | n/a | 0.003 | n/a | -2.12e-07 | n/a | n/a | n/a | 4.24e-03 | 2.24e-03 | -0.828 | 100.0% | n/a | 6 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | n/a | -8.08e-08 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | n/a | -1.72e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | n/a | -8.08e-08 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | n/a | -1.72e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | n/a | -8.08e-08 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | n/a | -1.72e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | n/a | -8.08e-08 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | n/a | -1.72e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | n/a | -1.71e-07 | n/a | n/a | n/a | 7.93e-04 | 1.46e-03 | n/a | n/a | n/a | 4 |

### QLIKE Diagnostics
| run | crisis_label | estimator | DM_p_EW_QLIKE | DM_p_MV_QLIKE | mean_qlike | substitution_fraction | no_iso_skip_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Aliased | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Constant-Correlation | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Ledoit-Wolf | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | OAS | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | SCM | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Tyler-Shrink | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | n/a | n/a | -7.622 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 0.050 | 4.3e-04 | -8.493 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 0.074 | 3.5e-04 | -8.743 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | n/a | n/a | -7.583 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | n/a | n/a | -1.351 | 75.0% | 25.0% |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Aliased | n/a | n/a | -5.882 | 100.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Constant-Correlation | n/a | n/a | -5.851 | 100.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Ledoit-Wolf | 7.6e-06 | 2.5e-07 | -6.622 | 100.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | OAS | 6.7e-05 | 1.8e-07 | -6.739 | 100.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | SCM | n/a | n/a | -5.882 | 100.0% | n/a |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | crisis_20200215_20200531 | Tyler-Shrink | n/a | n/a | -0.828 | 100.0% | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Aliased | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Constant-Correlation | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Ledoit-Wolf | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | OAS | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | SCM | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Tyler-Shrink | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Aliased | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Constant-Correlation | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Ledoit-Wolf | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | OAS | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | SCM | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Tyler-Shrink | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Aliased | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Constant-Correlation | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Ledoit-Wolf | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | OAS | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | SCM | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Tyler-Shrink | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Aliased | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Constant-Correlation | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Ledoit-Wolf | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | OAS | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | SCM | n/a | n/a | n/a | n/a | n/a |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Tyler-Shrink | n/a | n/a | n/a | n/a | n/a |

### What Limits Coverage
What limits coverage: no dominant rejection signals.

| run | edge_buffer | off_component_ratio | stability_fail | energy_floor | neg_mu |
| --- | --- | --- | --- | --- | --- |
| oneway_J5_solver-auto_est-cc_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-factor_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-lw_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-oas_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Highlights
- Average detection coverage across RC runs: 96.4%.
- Delta MSE (EW) < 0 in 45.2% of comparisons (median Delta MSE (EW) = 1.79e-08).
- 4 of 11 Diebold–Mariano tests show p < 0.05.
- 7 of 8 QLIKE DM tests show p < 0.05.
- Accepted detections substitute in 87.5% of evaluated windows.
- Median alignment angle across runs: 24.3° (mean 24.3°).
- 'no_isolated_spike' gate skipped 25.0% of windows on average (max 25.0%).

## Ablation Snapshot
No ablation sweep artifacts were detected for this gallery.

(no ablation sweep detected)



## Artifacts
- oneway_J5_solver-auto_est-cc_prep-none: tables: figures/smoke/oneway_J5_solver-auto_est-cc_prep-none/tables; plots: figures/smoke/oneway_J5_solver-auto_est-cc_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-cc_prep-none/run_meta.json
- oneway_J5_solver-auto_est-dealias_prep-none: tables: figures/smoke/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/smoke/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json
- oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531: tables: figures/smoke/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/tables; plots: figures/smoke/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/run_meta.json
- oneway_J5_solver-auto_est-factor_prep-none: tables: figures/smoke/oneway_J5_solver-auto_est-factor_prep-none/tables; plots: figures/smoke/oneway_J5_solver-auto_est-factor_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-factor_prep-none/run_meta.json
- oneway_J5_solver-auto_est-lw_prep-none: tables: figures/smoke/oneway_J5_solver-auto_est-lw_prep-none/tables; plots: figures/smoke/oneway_J5_solver-auto_est-lw_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-lw_prep-none/run_meta.json
- oneway_J5_solver-auto_est-oas_prep-none: tables: figures/smoke/oneway_J5_solver-auto_est-oas_prep-none/tables; plots: figures/smoke/oneway_J5_solver-auto_est-oas_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-oas_prep-none/run_meta.json
- oneway_J5_solver-auto_est-tyler_shrink_prep-none: tables: figures/smoke/oneway_J5_solver-auto_est-tyler_shrink_prep-none/tables; plots: figures/smoke/oneway_J5_solver-auto_est-tyler_shrink_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-tyler_shrink_prep-none/run_meta.json
