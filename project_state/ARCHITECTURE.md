---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Architecture

- **Data ingress & registries** — `src/io/wrds_connect.py`, `src/io/crsp_daily.py` pull WRDS CRSP daily returns; hashes are pinned in `data/registry.json` and factor digests in `data/factors/registry.json`. `tools/verify_dataset.py` guards inputs before runs.
- **Panel prep** — `experiments/daily/grouping.py`, `eval.clean`, and `eval.balance` align/winsorize/Huber-clip returns into balanced Week×Group panels; optional prewhitening via `experiments/prewhiten.py` and factor loaders in `baselines.factors`.
- **Base covariance layer** — `src/finance` (Ledoit/OAS/robust SCM in `shrinkage.py`/`robust.py`/`ledoit.py`; factor helpers in `factors.py`; portfolio utilities in `portfolio.py`/`portfolios.py`; design helpers in `design.py`) plus `src/baselines/covariance.py`/`factors.py` for evaluation baselines.
- **FJS detection & overlay** — `fjs.dealias`, `fjs.overlay` (detect_spikes/apply_overlay API), `fjs.mp` (MP edge & stability), `fjs.gating` (acceptance rules), `fjs.theta_solver`, `fjs.balanced`/`balanced_nested` (balanced MANOVA sums), `fjs.robust` (robust edges), `fjs.spectra` (spectral transforms).
- **Evaluation & risk metrics** — `evaluation.evaluate` (rolling metrics, alignment diagnostics), `evaluation.dm` (DM tests), `evaluation.factor` (observed-factor + POET-lite covariances), `finance.eval` (overlay-aware covariance evaluation), `finance.portfolios` (MV optimisation with skip/fail-loud solver handling), `plotting.utils` for diagnostics.
- **Experiment drivers** — daily evaluation (`experiments/eval/run.py`), weekly equity panel runner (`experiments/equity_panel/run.py`), synthetic harness (`experiments/synthetic/*.py`), ablations (`experiments/ablate/run.py`), ETF demo (`experiments/etf_panel/run.py`), and synthetic-oneway demo (`experiments/synthetic_oneway/run.py`).
- **Reporting** — `src/report` modules plus `tools/build_gallery.py`, `build_memo.py`, `build_brief.py`, `make_summary.py`, `summarize_rc_sanity.py`, `summarize_weekly_diagnostics.py`, `plot_rc_hist.py`. Outputs under `reports/` and `figures/rc/`.
- **Caching & metadata** — `meta/cache.py` for keyed cache files (`.cache/`), `meta/run_meta.py` for run provenance, `meta/runtime.py` for deterministic/throughput thread caps. Run manifests are written alongside outputs (`resolved_config.json`, `run.json`, `config_resolved.yaml`).
- **Orchestration** — `Makefile` exposes rc/rc-lite/rc-lite-sanity, synthetic sweeps, AWS dispatch (`aws:%` -> `scripts/aws_run.sh`), and gpt-bundle packaging. Targets call the Python entrypoints above.
