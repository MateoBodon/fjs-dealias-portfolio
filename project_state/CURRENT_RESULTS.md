# Current Results (as of 2025-12-08)

- **Latest RC-lite (DoW + Vol-state, deterministic, capped first 200 windows)** — `reports/rc-20251121/`
  - Edge mode Tyler, prewhiten FF5+MOM, window 126×21, top-60 assets (cap implied by README note “first 200 windows”).
  - Detection ≈4.32% (DoW), 4.33% (Vol); acceptance ~ detection; substitution ≈100% (artifact of cap).
  - ΔMSE(EW): +1.75e-13 (DoW), −1.05e-13 (Vol); ΔMSE(MV): −2.54e-14 (DoW), −8.64e-14 (Vol).
  - Regimes merged into `regime.csv`; summary tables in `reports/rc-20251121/summary/` and `metrics_summary.json`.
  - Memo/brief: `reports/memo.md`, `reports/brief.md` with timestamped copies under `reports/`.
- **Calibration defaults (HARNESS_TRIALS=800, deterministic)** — refreshed 2025-11-21
  - `calibration_defaults.json` selects SCM edge with energy_floor ≈0.108129 (target FPR 2%, power≈1.0 at μ∈{4,6,8}).
  - ROC figures in `reports/figures/roc_null.png`, `roc_power.png`; thresholds per edge-mode/p-bin in `calibration/edge_delta_thresholds.json`.
- **Synthetic harness outputs**
  - `reports/synthetic/null_harness/` and `reports/synthetic/power_harness/` store score tables and ROC sweeps (edge modes scm/tyler).
- **Data integrity**
  - `data/returns_daily.csv` sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`.
  - `data/factors/ff5mom_daily.csv` sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`.

### Gaps / caveats
- Nested design not included in the capped 2025-11-21 RC-lite; needs full-length rerun for production metrics.
- Crisis slices present historically (2020/2022 configs) but latest committed RC-lite omits them; crisis performance remains weaker (ΔMSE > 0 vs shrinkage per README examples).
- Ablation summary timed out in latest RC-lite (placeholder in gallery/memo); ablation grid needs rerun or pruning.
