# Release Candidate Memo - rc

This memo summarizes the latest smoke and crisis release-candidate runs, highlighting detection coverage, variance improvements, and statistical significance.

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
| run | crisis_label | estimator | detection_rate | delta_mse_ew | delta_mse_mv | dm_p_ew | dm_p_mv | n_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Aliased | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Constant-Correlation | 1 | -7.512e-08 | -8.08e-08 | nan | nan | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Ledoit-Wolf | 1 | -3.583e-07 | -1.71e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | OAS | 1 | -3.7e-07 | -1.708e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | SCM | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-cc_prep-none | full_oneway_J5_solver-auto_est-cc_prep-none | Tyler-Shrink | 1 | 0.08601 | -1.709e-07 | 0.07386 | nan | 4 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | 1 | 6.668e-07 | -4.163e-11 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 1 | 6.97e-07 | 1.078e-10 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 1 | -1.047e-06 | 8.464e-07 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 1 | -1.11e-06 | 8.58e-07 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | 1 | 6.668e-07 | -4.163e-11 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 1 | 0.2574 | 1.035e-06 | 0.01372 | nan | 3 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Aliased | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Constant-Correlation | 1 | -7.512e-08 | -8.08e-08 | nan | nan | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Ledoit-Wolf | 1 | -3.583e-07 | -1.71e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | OAS | 1 | -3.7e-07 | -1.708e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | SCM | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-factor_prep-none | full_oneway_J5_solver-auto_est-factor_prep-none | Tyler-Shrink | 1 | 0.08601 | -1.709e-07 | 0.07386 | nan | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Aliased | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Constant-Correlation | 1 | -7.512e-08 | -8.08e-08 | nan | nan | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Ledoit-Wolf | 1 | -3.583e-07 | -1.71e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | OAS | 1 | -3.7e-07 | -1.708e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | SCM | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-lw_prep-none | full_oneway_J5_solver-auto_est-lw_prep-none | Tyler-Shrink | 1 | 0.08601 | -1.709e-07 | 0.07386 | nan | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Aliased | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Constant-Correlation | 1 | -7.512e-08 | -8.08e-08 | nan | nan | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Ledoit-Wolf | 1 | -3.583e-07 | -1.71e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | OAS | 1 | -3.7e-07 | -1.708e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | SCM | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-oas_prep-none | full_oneway_J5_solver-auto_est-oas_prep-none | Tyler-Shrink | 1 | 0.08601 | -1.709e-07 | 0.07386 | nan | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Aliased | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Constant-Correlation | 1 | -7.512e-08 | -8.08e-08 | nan | nan | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Ledoit-Wolf | 1 | -3.583e-07 | -1.71e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | OAS | 1 | -3.7e-07 | -1.708e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | SCM | 1 | 1.79e-08 | -1.715e-07 | nan | nan | 4 |
| oneway_J5_solver-auto_est-tyler_shrink_prep-none | full_oneway_J5_solver-auto_est-tyler_shrink_prep-none | Tyler-Shrink | 1 | 0.08601 | -1.709e-07 | 0.07386 | nan | 4 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Aliased | 0 | 0 | 0 | nan | nan | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 0 | 1.845e-07 | 4.033e-12 | nan | nan | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 0 | 3.671e-09 | 5.573e-08 | nan | nan | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | OAS | 0 | 2.075e-08 | 4.131e-08 | nan | nan | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | SCM | 0 | 1.526e-07 | -4.278e-13 | nan | nan | 24 |
| nested_J5_solver-auto_est-dealias_prep-none | full_nested_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 0 | 0.1287 | 2.594e-06 | 0 | nan | 24 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | 1 | 6.668e-07 | -4.163e-11 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 1 | 6.97e-07 | 1.078e-10 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 1 | -1.047e-06 | 8.464e-07 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 1 | -1.11e-06 | 8.58e-07 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | 1 | 6.668e-07 | -4.163e-11 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 1 | 0.2574 | 1.035e-06 | 0.01372 | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Aliased | 1 | 6.668e-07 | -4.163e-11 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Constant-Correlation | 1 | 6.97e-07 | 1.078e-10 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Ledoit-Wolf | 1 | -1.047e-06 | 8.464e-07 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | OAS | 1 | -1.11e-06 | 8.58e-07 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | SCM | 1 | 6.668e-07 | -4.163e-11 | nan | nan | 3 |
| oneway_J5_solver-auto_est-dealias_prep-none | full_oneway_J5_solver-auto_est-dealias_prep-none | Tyler-Shrink | 1 | 0.2574 | 1.035e-06 | 0.01372 | nan | 3 |

## Highlights
- Average detection coverage across RC runs: 85.7%.
- ΔMSE(EW) < 0 in 38.9% of comparisons (median ΔMSE(EW) = 1.79e-08).
- 4 of 9 Diebold–Mariano tests show p < 0.05.

## Artifacts
- oneway_J5_solver-auto_est-cc_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-cc_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-cc_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-cc_prep-none/run_meta.json
- oneway_J5_solver-auto_est-dealias_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2020/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json; tables: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_crisis_2022/oneway_J5_solver-auto_est-dealias_prep-none/run_meta.json
- oneway_J5_solver-auto_est-factor_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-factor_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-factor_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-factor_prep-none/run_meta.json
- oneway_J5_solver-auto_est-lw_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-lw_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-lw_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-lw_prep-none/run_meta.json
- oneway_J5_solver-auto_est-oas_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-oas_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-oas_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-oas_prep-none/run_meta.json
- oneway_J5_solver-auto_est-tyler_shrink_prep-none: tables: figures/rc/oneway_J5_solver-auto_est-tyler_shrink_prep-none/tables; plots: figures/rc/oneway_J5_solver-auto_est-tyler_shrink_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-tyler_shrink_prep-none/run_meta.json
- nested_J5_solver-auto_est-dealias_prep-none: tables: figures/rc/nested_J5_solver-auto_est-dealias_prep-none/tables; plots: figures/rc/nested_J5_solver-auto_est-dealias_prep-none/plots; run_meta: /Users/mateobodon/Documents/Programming/Projects/fjs-dealias-portfolio/experiments/equity_panel/outputs_nested_smoke/nested_J5_solver-auto_est-dealias_prep-none/run_meta.json
