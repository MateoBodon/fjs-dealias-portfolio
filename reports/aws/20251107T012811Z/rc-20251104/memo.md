# FJS De-aliasing Overlay RC (2025-11-04)

## Run Synopsis
- **Date:** 2025-11-04 (EST)
- **Input panel:** `data/returns_daily.csv` (2010-01-05 → 2023-12-29)
- **Replicate designs:**
  - **DoW (dow-bounded):** prewhiten FF5+MOM, shrinker `rie`
  - **Vol-state (vol-bounded):** no prewhiten, shrinker `oas`
- **Window / horizon:** 126-day estimation, 21-day holdout
- **Asset cap:** first 80 tickers (alphabetical)
- **Gate:** strict (Tyler edge) with calibrated δ-frac (α=2%) and `gate_delta_frac_min=0.02`
- **Baselines logged:** sample, RIE, LW, OAS, CC, QuEST, EWMA, factor (observed FF5+MOM), POET-lite

## Evaluation Highlights
- **Detection cadence:**
  - DoW detection rate 3.4 % (full), 4.6 % (crisis); substitution fraction ≤5.6 %.
  - Vol-state detection rate 2.9 % (full), 4.1 % (crisis); substitution fraction ≤3.9 %.
- **Edge quality:** Median edge margin ≥0.35 with isolation share ≥0.65 across both designs.
- **Overlay vs baseline:**
  - ΔMSE vs RIE negative in calm bins, slightly positive during crisis windows (overlay underperforms in late-2020 regime snippets).
  - VaR95 coverage error within ±1 % for overlay and RIE; ES squared error improvements modest (<3 %).
- **Factor telemetry:** DoW prewhitening R² mean ≈0.41, |β| mean 0.27. Vol-state runs logged mode `off` as expected.

## Calibration Summary
- `calibration/edge_delta_thresholds.json` — synthetic surrogate (α=0.02) for SCM & Tyler modes, enforcing FPR ≤2 % at p/n≈0.9.
- Δ thresholds consumed via `--gate-delta-calibration` for both RC jobs; strict gate rejects non-admissible roots & sub-threshold stability.

## Artifacts
- `reports/rc-20251104/dow-bounded/`
- `reports/rc-20251104/vol-bounded/`
- Smoke hooks: `reports/rc-20251104/{dow,vol}-smoke-top80/`
- Overlay toggle & diagnostics per regime plus `regime.csv`, `prewhiten_diagnostics.csv`, `resolved_config.json` in each directory.

## Next Actions
1. Rerun full synthetic calibration (≥600 trials) when longer wall-clock budget available to replace surrogate thresholds.
2. Inspect crisis ΔMSE under-performance; consider relaxing `gate_delta_frac_max` or adapting alignment thresholds.
3. Regenerate memo gallery (`make gallery`) and RC memo packaging once crisis tuning stabilises.
