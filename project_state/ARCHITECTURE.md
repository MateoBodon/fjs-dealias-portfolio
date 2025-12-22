---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Architecture

- **Data ingress & registries** — `src/io/wrds_connect.py`, `src/io/crsp_daily.py` pull WRDS CRSP daily returns; `src/data/registry.py` and `data/registry.json`/`data/factors/registry.json` track hashes; `tools/verify_dataset.py` enforces inputs.
- **Panel prep & preprocessing** — `eval.clean`/`eval.balance` apply NaN policies and build balanced windows; `experiments.daily.grouping` builds group labels; `experiments.prewhiten` applies factor prewhitening; `src/data/panels.py` stores balanced Week×Day panels and manifests.
- **Base covariance + factors** — `src/finance` (Ledoit/OAS/robust SCM, factor covariance, portfolio helpers) plus `src/baselines` for evaluation baselines; factor data handled by `src/finance.factors` and `experiments.prewhiten`.
- **FJS detection + overlay** — `src/fjs` modules: `overlay.py` (detect_spikes/apply_overlay), `dealias.py` (spectral transform), `mp.py` (MP edge), `gating.py` (acceptance rules), `balanced*.py` (MANOVA sums), `theta_solver.py`, `spectra.py`, `robust.py`.
- **Evaluation + risk metrics** — `src/evaluation.evaluate` (rolling metrics + alignment), `src/evaluation.dm` (DM tests), `src/evaluation.factor` (observed-factor + POET-lite); `src/finance.eval` (covariance evaluation) and `src/finance.portfolios` (MV optimizer with skip/fail-loud solver handling).
- **Experiment drivers** — daily evaluation `experiments/eval/run.py`, weekly equity panel `experiments/equity_panel/run.py`, synthetic harness `experiments/synthetic/*.py`, ablations `experiments/ablate/run.py`, ETF demo `experiments/etf_panel/run.py`, daily smoke `experiments/daily/run.py`, synthetic-oneway `experiments/synthetic_oneway/run.py`.
- **Reporting & summaries** — `src/report` plus `tools/make_summary.py`, `tools/summarize_rc_sanity.py`, `tools/summarize_weekly_diagnostics.py`, `tools/build_gallery.py`, `tools/build_memo.py`, `tools/build_brief.py`, `tools/plot_rc_hist.py`.
- **Caching & provenance** — `src/meta` (`cache.py`, `run_meta.py`, `runtime.py`, `completeness.py`) plus per-run `resolved_config.json`/`config_resolved.yaml`/`run.json`.
- **Orchestration** — `Makefile` targets (rc/rc-lite/rc-lite-sanity, synthetic sweeps, gpt-bundle) call the entrypoints above; see `project_state/_generated/make_targets.txt`.
