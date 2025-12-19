---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Current Results (latest validated drops)

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

Older runs (AWS RCs, sensitivity sweeps, prewhiten studies) remain catalogued in `reports/rc-20251113/`, `reports/rc-sensitivity/`, and `reports/aws/`; see PROGRESS.md for provenance.
