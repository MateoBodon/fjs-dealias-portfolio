# Runbook

## Smoke vs. full equity runs

- Smoke run (≈5 min, 6+1 weeks, 80 assets):
  ```bash
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py \
      --config experiments/equity_panel/config.smoke.yaml \
      --no-progress --workers $(python -c 'import os;print(os.cpu_count() or 4)') \
      --assets-top 80 --stride-windows 4 --resume --cache-dir .cache \
      --precompute-panel --drop-partial-weeks --estimator oas
  ```
  Artifacts land in `experiments/equity_panel/outputs_smoke/<design>_J*_solver-*_est-*_prep-*/`.
- Full run (≈90 min, 156+4 weeks): drop the stride/top overrides and let `--estimator dealias`.

## Test suites

- `make test-fast` → `pytest -m unit` (deterministic micro-tests; keep data to ≤12 assets, ≤6 weeks).
- `make test-integration` → `pytest -m integration` (smoke-sized multi-module flows).
- `make test-slow` → `pytest -m slow` (long synthetic studies, optional locally).
- `make test-all` → union of unit + integration (default CI footprint). Mark multi-minute jobs as `heavy`.

## New estimator switches

- `--estimator {aliased,dealias,lw,oas,cc,factor,factor_obs,poet,tyler_shrink}` tags the run, cache entries, and `run_meta`.
- `--factor-csv tests/data/factors_tiny.csv` enables the observed-factor covariance; the CSV must be wide, date-indexed. `factor_obs` auto-skips (with a warning) if factors are absent, while `poet` needs no extra inputs.
- Shrinkage benchmarks (LW, OAS, constant-correlation, Tyler) and the factor estimator all appear in `metrics_summary.csv`; Diebold–Mariano columns (`dm_stat_*`, `dm_p_*`) compare them to the de-aliased baseline.

## Robust preprocessing toggles

- `--winsorize q` applies column-wise clipping to the empirical `[q, 1-q]` quantiles (mutually exclusive with Huber).
- `--huber c` clips at `median ± c·MAD` per column; use when thin tails help guard ablations.
- Preprocess selections become part of cache keys, panel manifests, artifact directories, and `run_meta.preprocess_flags`.
- Pair with `--estimator tyler_shrink` to use the Tyler–ridge covariance in evaluation and DM tables.

## Edge-mode overrides

- `--edge-mode {scm,tyler,huber}` scales the MP edge before applying the δ/δ_frac buffers. Each window records both `edge_scm` and `edge_selected` in `detection_summary.csv` and `summary.json` logs the chosen mode.
- `--edge-huber-c 1.5` tunes the Huber scatter threshold when `--edge-mode huber`.
- Memo lines include `[edge=<mode>]` badges; nested runs that bail solely due to `no_isolated_spike` are flagged with “nested scope de-scoped”.

## Min-variance regularisation and turnover

- `--minvar-ridge 1e-3` adds λ²I to the min-var covariance; tweak with `--minvar-box lo,hi` (defaults 0,0.05).
- Turnover diagnostics (`*_turnover`, `*_turnover_cost`) are written per window.
- Apply transaction costs via `--turnover-cost 25` (basis points).

## Summaries

- `python tools/summarize_run.py <run_dir>` prints estimator MSE breakdowns plus DM p-values.
- `tools/list_runs.py` remains the index of archived runs.

## Clean outputs

- Inspect pending moves: `python tools/clean_outputs.py --dry-run`
- Delete legacy artifacts once verified: `python tools/clean_outputs.py --purge`

## Build gallery

- Build tables/plots for configured runs: `make gallery`
- Inspect generated assets: `ls figures/smoke/<run_tag>/{tables,plots}`

## Release Candidate

- Generate RC runs, gallery, and memo: `make rc`
- Review outputs: `ls figures/rc/<run_tag>` and `less reports/memo.md`
- Extend coverage by editing `experiments/equity_panel/config.rc.yaml` (add estimators or crisis configs) before rerunning `make rc`.

## Calibration (Oct 2025)

- **Smoke + crisis reruns.** `PYTHONPATH=src python3 experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml --no-progress --precompute-panel --drop-partial-weeks --estimator dealias` and the matching `--crisis 20200215:20200531` rerun now write gating telemetry (`windows_substituted`, `skip_reasons`) plus alignment angles in `detection_summary.csv`. With the current defaults the smoke slice substituted 3/4 windows (one `no_isolated_spike` skip, median alignment ≈17°) and the 2020 crisis slice substituted all six windows (median alignment ≈31°).
- **Gallery + memo regeneration.** `python3 tools/build_gallery.py --config experiments/equity_panel/config.gallery.yaml` and `python3 tools/build_memo.py --config experiments/equity_panel/config.gallery.yaml` refresh the plots/tables; look for the new `alignment_angles.png` plot per run and the memo’s QLIKE/alignment panel.
- **Acceptance sweep.** `PYTHONPATH=src python3 experiments/equity_panel/sweep_acceptance.py --design oneway --estimators dealias lw oas cc tyler --grid default` populates `experiments/equity_panel/sweeps/` with tagged run directories, `sweep_summary.csv`, and the overview heatmaps `E5_detection_rate.png` / `E5_mse_gain.png`. The grid `{δ_frac, ε} = {0.01,0.02} × {0.02,0.03}` with `η ∈ {0.4,0.6}` and `a_grid ∈ {90,120}` produced indistinguishable detection and ΔMSE profiles, so we retained the guardrail values (δ_frac = 0.02, ε = 0.03, η = 0.4) and standardised on `a_grid = 120`.
- **Run aggregation.** `python3 tools/aggregate_runs.py --inputs "experiments/equity_panel/outputs*" --out reports/aggregate_summary.csv --tex-out reports/aggregate_summary.tex` concatenates the latest slices; the table includes both MSE and QLIKE columns for quick comparisons.
- **Defaults snapshot.** `experiments/equity_panel/config.yaml` and `config.smoke.yaml` now carry `gating` defaults (`enable: true`, `q_max: 2`, `require_isolated: true`) alongside `alignment_top_p: 3`.
