---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Module Summaries

- **src/fjs** — spectral de-alias overlay. `dealias.py` (core transform + admissibility), `overlay.py` (detect_spikes/apply_overlay), `mp.py` (MP edge), `gating.py` (acceptance), `balanced.py` / `balanced_nested.py` (balanced MANOVA sums), `theta_solver.py` (t-vector solver), `spectra.py` (eigen transforms), `robust.py` (robust edge scaling).
- **src/finance** — base covariance + portfolio layer. `shrinkage.py`/`ledoit.py` (LW/OAS/RIE wrappers), `robust.py` (Tyler/Huber SCM), `factors.py` (factor loading helpers), `returns.py`/`io.py`/`loader.py` (data alignment), `design.py` (group labels), `portfolio.py` + `portfolios.py` (EW, min-var, box constraints, turnover, solver skip/fail-loud), `eval.py` (overlay-aware covariance evaluation).
- **src/baselines** — covariance baselines (`covariance.py`: cc, EWMA, LW, OAS, QUEST, RIE) and factor loading/prewhitening helpers (`factors.py`).
- **src/evaluation** — `evaluate.py` (rolling metrics, alignment diagnostics, q-like losses), `dm.py` (Diebold–Mariano tests incl. flip-set), `factor.py` (observed-factor + POET-lite covariances).
- **src/eval** — `clean.py` (NaN policy / winsor/Huber), `balance.py` (balanced panel construction), reused by `experiments/eval/run.py` and tests.
- **src/meta** — `cache.py` (on-disk memoisation), `completeness.py` (run completeness checks), `run_meta.py` (git/code signature hashes), `runtime.py` (EXEC_MODE thread caps + seeds).
- **src/report** — `gather.py` (load metrics/diagnostics), `tables.py` (summary tables), `plots.py` (histograms, DM plots, completeness badges). Used by gallery/memo builders.
- **src/plotting** — `utils.py` (common plotting helpers used in tests and tools).
- **src/io** — `wrds_connect.py` (WRDS DB session helper), `crsp_daily.py` (fetch/format CRSP daily returns).
- **experiments/equity_panel** — `run.py` weekly group runner (DoW/vol/nested), configs `config.yaml`, `config.rc.yaml`, `config.smoke.yaml`, `config.nested.smoke.yaml`, `config.ablation.smoke.yaml`, `config.gallery.yaml`, `config.crisis.2020.yaml`, `config.crisis.2022.yaml`, `config.nested.crisis.2020.yaml`. Produces `outputs_*` dirs with `detection_summary.csv`, `weekly_diagnostics.md`, spectra/edge plots.
- **experiments/eval** — `run.py` daily evaluation CLI (EW/MV, VaR/ES, DM), `config.py` (layered config merge), `diagnostics.py`, `inject_spike.py`, `sensitivity.py`, `config.yaml` + `thresholds.json` defaults.
- **experiments/synthetic** — `null.py`, `power.py`, `power_null.py`, `calibrate_thresholds.py`, `harness_utils.py`, `nested_killtest.py`, `config.nested.killtest.yaml` for ROC/null/power and acceptance calibration.
- **experiments/ablate** — `run.py` drives ablation grid defined in `ablation_matrix.yaml` / `ablation_matrix_tiny.yaml`.
- **experiments/daily** — `run.py` small daily smoke, `grouping.py` group label helpers.
- **experiments/prewhiten.py** — factor prewhitening CLI used by eval/equity runners.
- **experiments/synthetic_oneway** — S1/S3/S4/S5 harness and plots.
- **experiments/etf_panel** — `run.py` ETF panel demo mirroring equity eval defaults.
- **tools/** — ops/reporting utilities: `verify_dataset.py`, `update_registry.py`, `list_runs.py`, `aggregate_runs.py`, `run_monitor.py`, `clean_outputs.py`, `shard_grid.py`/`reduce_calibration.py`, `make_summary.py`, `summarize_rc_sanity.py`, `summarize_weekly_diagnostics.py`, `build_gallery.py`, `build_memo.py`, `build_brief.py`, `plot_rc_hist.py`, `summarize_run.py`, `prewhiten_effect.py`, `generate_project_state.py`.
- **tests/** — coverage for fjs math/gating, synthetic harness, eval/equity runners, reporting, registries, MV solver skip/fail-loud, gpt-bundle packaging.
