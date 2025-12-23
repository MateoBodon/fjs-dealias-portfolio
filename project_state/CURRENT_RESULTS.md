---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Current Results (latest validated drops)

- **2025-12-22 — Daily DoW paper-v1 (ticket-07, git 2cb5bfd)**  
  - Deterministic daily DoW run (`experiments/eval/config.paper_v1.yaml`, FF5+MOM prewhiten) with uncapped windows: `cap_active=false`, `window_coverage=1.0`, `n_effective=1749` (full regime).  
  - Full-regime detection_rate_mean ≈ 4.16% (1751/1774 windows); window drops logged as `holdout_empty: 115` (excluded from planning, not treated as caps).  
  - Performance deltas (full regime): EW ΔQLIKE ≈ −0.06719 (ΔMSE ≈ +2.64e-11), MV ΔQLIKE ≈ −0.03576 (ΔMSE ≈ −6.65e-13).  
  - Artifacts: `reports/rc-ticket-07-20251222_183800/summary/{summary_perf.csv,summary_detection.csv,overlay_forensics.csv,limitations.md,advisor_snapshot.md}`; run dir `reports/rc-ticket-07-20251222_183800/dow-paper-v1/`.

- **2025-12-22 — Daily DoW paper-v1 (ticket-06, git 8a5579b)**  
  - Deterministic daily DoW run (`experiments/eval/config.paper_v1.yaml`, FF5+MOM prewhiten) with uncapped windows: `cap_active=false`, `window_coverage=1.0`, `n_effective=1749` (full regime).  
  - Full-regime detection_rate_mean ≈ 4.16% (1751/1774 windows); window drops logged as `holdout_empty: 115` (excluded from planning, not treated as caps).  
  - Performance deltas (full regime): EW ΔMSE ≈ +2.64e-11 (harmful), MV ΔMSE ≈ −6.65e-13 (slight improvement).  
  - Artifacts: `reports/rc-ticket-06-20251222_063304/summary/{summary_perf.csv,summary_detection.csv,overlay_forensics.csv,limitations.md}`; run dir `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`.

- **2025-12-20 — Nested synthetic calibration (ticket-10, git e6e7982)**  
  - Synthetic nested (p=200, years=2, weeks 6–8, reps=5, tyler, delta=0.35, delta_frac=0.05): null detections 0/220 → FPR 0 with Wilson hi 0.017; power 1.0 on moderate/strong.  
  - Calibration written to `calibration/nested_edge_delta_thresholds.json` with run metadata; nested configs now point to this file; lookup is design-aware.
  - (ticket-14 fixup) Calibration artifact now includes config hash + operating_points and enforces design-specific lookup; tiny deterministic nested smoke (max_windows=3) on WRDS data skipped 3/3 windows with `skip_reason=calibration_missing_p_T` (p≈188, T=70/80) and delta_frac_used=0.008 fallback—operating point unchanged.
- **2025-12-19 — MV solver missing-proof (ticket-08, git a4451969)**
  - Commands: `make test-fast`; `python -m experiments.eval.run ... --mv-solver cvxpy` and forced-missing run with `FJS_FORCE_MISSING_CVXPY=1 --mv-skip-on-missing-solver`.
  - Outcomes: Normal run `reports/eval-smoke-ticket08-proof/normal/metrics_detail.csv` shows MV rows `skipped=False`, `solver_status=optimal`; forced-missing run `.../missing-skip/metrics_detail.csv` shows `skipped=True`, `skip_reason=missing_solver`, empty weights; diagnostics propagate `solver_used`/`solver_status`.
- **2025-12-19 — Weekly gating diagnostics (ticket-07, git 2e0fd573b5)**
  - Real DoW smoke (2023Q1, window=6, horizon=1): detection_rate=0.75 (3/4) with one `no_isolated_spike`; guardrail tallies dominated by `guard_other`=1148 (`experiments/equity_panel/outputs_smoke_ticket07_20251219_173231/weekly_diagnostics.md`).
  - Synthetic micro smoke: detection_rate=0, `skip_reason=diagnostic_failure` across all windows (`experiments/equity_panel/outputs_ticket07_synth_20251219_173231/weekly_diagnostics.md`).
- **2025-12-19 — rc-lite-sanity completeness refresh (ticket-05, git 03d4c03c)**
  - Deterministic DoW/vol daily eval (top-50, 60×10): detection_rate≈0.055 (DoW) / 0.052 (vol); overlay_effect harmful (ΔMSE > 0), percent_changed≈100%; completeness JSON emitted under `reports/rc-20251219-sanity-20251219_050735/summary/`.
  - Weekly DoW + nested remain zero-acceptance; completeness surfaced in `summary_sanity.json` and `limitations.md`.
- **2025-11-21 — Latest full RC-lite (deterministic)**
  - DoW/vol (Tyler edge, FF5+MOM, top-60, first 200 windows): detection≈4.3%, acceptance≈detection, percent_changed≈100%, ΔMSE(EW)=+1.75e-13, ΔMSE(MV)=−2.54e-14. Artefacts in `reports/rc-20251121/` (`metrics_summary.json`, `run_manifest.json`).

Older RC-lite (2025-11-21) and prewhitening/vol acceptance studies remain under `reports/rc-20251121/` and `reports/rc-20251113/`; see PROGRESS.md for provenance.
