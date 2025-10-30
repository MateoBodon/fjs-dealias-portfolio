# Release Candidate Memo - rc

This memo synthesizes smoke and crisis release-candidate runs, highlighting estimator deltas, coverage bottlenecks, and ablation sensitivities.

## Run Selection
- **full_oneway_J5_solver-auto_est-cc_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-factor_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-lw_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-oas_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-tyler_shrink_prep-none** (design=oneway, J=5, period=2023-01-01 → 2023-03-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_nested_J5_solver-auto_est-dealias_prep-none** (design=nested, J=5, period=2022-01-01 → 2023-12-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2020-02-01 → 2020-05-31) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink
- **full_oneway_J5_solver-auto_est-dealias_prep-none** (design=oneway, J=5, period=2022-09-01 → 2022-11-30) — estimators: Aliased, Constant-Correlation, Ledoit-Wolf, OAS, SCM, Tyler-Shrink

## Key Results
Values are measured versus the de-aliased baseline; negative Delta MSE indicates an improvement.

| run | crisis_label | estimator | detection_rate | delta_mse_ew | CI_EW | DM_p_EW | delta_mse_mv | CI_MV | DM_p_MV | edge_margin_median | edge_margin_IQR | n_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Aliased | 0.0% | 0.00e+00 | [-0.00e+00, -0.00e+00] | n/a | 0.00e+00 | [-0.00e+00, -0.00e+00] | n/a | n/a | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 0.0% | 1.85e-07 | n/a | n/a | 4.03e-12 | n/a | n/a | n/a | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 0.0% | 3.67e-09 | [-7.40e-08, 1.57e-07] | n/a | 5.57e-08 | [5.17e-08, 5.96e-08] | n/a | n/a | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | OAS | 0.0% | 2.07e-08 | n/a | n/a | 4.13e-08 | n/a | n/a | n/a | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | SCM | 0.0% | 1.53e-07 | n/a | n/a | -4.28e-13 | n/a | n/a | n/a | n/a | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 0.0% | 0.129 | n/a | 0.0e+00 | 2.59e-06 | n/a | n/a | n/a | n/a | 24 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | -8.08e-08 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | -1.72e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | -8.08e-08 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | -1.72e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | -8.08e-08 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | -1.72e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | -8.08e-08 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | -1.72e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | -8.08e-08 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | -1.72e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Aliased | 100.0% | 1.79e-08 | [-1.12e-08, -1.12e-08] | n/a | -1.72e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Constant-Correlation | 100.0% | -7.51e-08 | n/a | n/a | -8.08e-08 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Ledoit-Wolf | 100.0% | -3.58e-07 | [-3.69e-07, -3.69e-07] | n/a | -1.71e-07 | [-1.71e-07, -1.71e-07] | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | OAS | 100.0% | -3.70e-07 | n/a | n/a | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | SCM | 100.0% | 1.79e-08 | n/a | n/a | -1.72e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Tyler-Shrink | 100.0% | 0.086 | n/a | 0.074 | -1.71e-07 | n/a | n/a | 7.93e-04 | 1.46e-03 | 4 |

### What Limits Coverage
What limits coverage: no dominant rejection signals.

| run | edge_buffer | off_component_ratio | stability_fail | energy_floor | neg_mu |
| --- | --- | --- | --- | --- | --- |
| nested_J5_solver-auto_est-dealias_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-cc_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-dealias_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-factor_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-lw_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-oas_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Highlights
- Average detection coverage across RC runs: 85.7%.
- Delta MSE (EW) < 0 in 42.9% of comparisons (median Delta MSE (EW) = 1.79e-08).
- 1 of 7 Diebold–Mariano tests show p < 0.05.

## Ablation Snapshot
No ablation sweep artifacts were detected for this gallery.

(no ablation sweep detected)



## Artifacts
- oneway_J5_solver-auto_est-cc_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-cc_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-cc_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-cc_prep-none/run_meta.json
- oneway_J5_solver-auto_est-dealias_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2020/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2022/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json
- oneway_J5_solver-auto_est-factor_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-factor_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-factor_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-factor_prep-none/run_meta.json
- oneway_J5_solver-auto_est-lw_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-lw_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-lw_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-lw_prep-none/run_meta.json
- oneway_J5_solver-auto_est-oas_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-oas_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-oas_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-oas_prep-none/run_meta.json
- oneway_J5_solver-auto_est-tyler_shrink_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-tyler_shrink_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-tyler_shrink_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-tyler_shrink_prep-none/run_meta.json
- nested_J5_solver-auto_est-dealias_prep-none: tables: figures/rc/nested_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/nested_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_nested_smoke/nested_J5_solver-auto_est-dealias_prep-none/run_meta.json
