# fjs-dealias-portfolio

Robust variance forecasting over balanced equity panels, with tooling to explore alternative estimators, crisis windows, and reporting assets. The project contains the de-aliased (FJS) covariance pipeline, supporting shrinkage baselines, preprocess toggles, and an automated gallery/memo workflow for review.

---

## Current Status — 2025-11-05

- **Balanced windows & NaN hygiene**: daily evaluation now enforces replicate balancing and missing-data caps with per-window telemetry (logs written alongside diagnostics).
- **Calibration**: replicate-aware synthetic sweep added; run via `scripts/run_calibration.sh` (writes `calibration/edge_delta_thresholds.json`). Latest long-form sweep is in-progress on external compute; prior artefacts remain under `calibration/`.
- **Latest RC artefacts**: refreshed reports live in `reports/rc-20251103/` (full/calm/crisis CSVs, `overlay_toggle.md`). Memo/run manifest for the upcoming RC will be generated after Steps 5–6.
- **Next milestones**: (1) integrate WRDS FF5+UMD factor loader (`src/io/wrds_factors.py`) and expose `--prewhiten` path in the runners, (2) tighten MP-edge bracketing & overlay stability gates, (3) produce bounded WRDS RC + memo. Track the sprint focus in `AGENTS.md`.

Data footprint (local): WRDS snapshots under `data/wrds/*.parquet` (preferred source), sample RC outputs under `reports/rc-20251103/`. Figures are generated on demand into `figures/` (gitignored).

---

## 1. Prerequisites & installation

| Requirement | Notes |
| --- | --- |
| Python ≥3.11 | Development is tested on 3.11/3.12. |
| C build tools | Needed for scientific Python packages (NumPy, SciPy). |
| `make`, `git` | Used for the convenience targets and CI workflows. |

```bash
git clone https://github.com/MateoBodon/fjs-dealias-portfolio.git
cd fjs-dealias-portfolio
python -m venv .venv
source .venv/bin/activate
make setup  # installs the package in editable mode with dev extras
```

Optional data: `experiments/equity_panel/config*.yaml` expect `data/returns_daily.csv`. Adjust the config(s) if you want to point at a different dataset.

---

## Cloud compute runner (AWS EC2)

- **Resources.** Region `us-east-1`, instance `i-075b6e3853fe2349e` (`ec2-3-236-225-54.compute-1.amazonaws.com`), SSH user `ubuntu`, key `~/.ssh/mateo-us-east-1-ec2-2025`, bucket `fjs-artifacts-mab-prod`, IAM role `EC2AdminRole`.
- **Environment.** Micromamba lives under `~/.local/share/mamba`; the `fjs` env (Python 3.11) already includes NumPy, SciPy, pandas, pyarrow, scikit-learn, boto3, s3fs, dask, ray, tqdm, click, etc. Thread caps (`OMP/MKL/OPENBLAS/NUMEXPR=1`) are exported via `~/.bashrc`, and the repo is cloned at `~/fjs-dealias-portfolio`.
- **Provisioning.** `scripts/aws_provision.sh` now defaults to `INSTANCE_TYPE=c7a.32xlarge`; override via environment variables if you need a lighter host. The script installs toolchains, pins BLAS threads to 1, and ensures micromamba + the `fjs` env are ready for rsync-driven jobs.
- **Usage.** Configure the local AWS CLI profile `fjs-prod`, confirm the SSH key permissions (`chmod 400 ~/.ssh/mateo-us-east-1-ec2-2025`), then connect with `ssh -i ~/.ssh/mateo-us-east-1-ec2-2025 ubuntu@ec2-3-236-225-54.compute-1.amazonaws.com`. Run jobs with `micromamba run -n fjs python experiments/eval/run.py ...` (or activate the env) and push artefacts to `s3://fjs-artifacts-mab-prod/reports/`.
- **Documentation.** Step-by-step cloud procedures—preflight checks, S3 hardening (AES256 + lifecycle), smoke testing, and report uploads—are captured in `docs/CLOUD.md`. The bucket also stores dated setup summaries (`docs/fjs-cloud-setup-pre-YYYY-MM-DD.md`, `docs/fjs-ec2-setup-YYYY-MM-DD.md`) for auditing.
- **Telemetry.** AWS runs automatically wrap commands with `tools/run_monitor.py`, writing `metrics.jsonl` (resource samples), `metrics_summary.json` (avg/peak CPU/memory/IO, runtime), and `progress.jsonl` (parsed progress events + ETA). Tune sampling via `MONITOR_INTERVAL=<seconds>` when calling `make aws:<target>`.
- **Instance families.** Set `INSTANCE_FAMILY={c7a|c7i|auto}` before running `scripts/aws_provision.sh`. In `auto` mode the script inspects the latest `reports/aws/bench/*.json` artefacts (see `scripts/bench_linalg.py` below) and chooses the faster family, falling back to `c7a` when no benchmarks are present. Override the size with `INSTANCE_SIZE=16xlarge` (defaults to `32xlarge`).
- **Benchmarks.** `scripts/bench_linalg.py` times 100× `numpy.linalg.eigh` at `p∈{100,200,300}` with the configured thread cap, writing metadata to `reports/aws/bench/<timestamp>/bench.json`. Run it locally via `make bench-linalg` or remotely with `make aws:bench-linalg` before switching instance families.

---

## 2. Data & Reproducibility

- **Daily input.** `experiments/equity_panel/run.py` looks for `data/returns_daily.csv` with columns `date,ticker,ret`. The repository ships a compact sample (Sharadar US equities through 2024-12-31) so smoke tests work offline.
- **Regenerating a mini sample.** If the returns file is missing, the runner will synthesise `data/prices_daily.csv` and derive returns on first use. To refresh the bundled sample explicitly, convert `data/prices_sample.csv` via:

  ```bash
  python - <<'PY'
  import pandas as pd
  from finance.io import load_prices_csv, to_daily_returns

  prices = load_prices_csv("data/prices_sample.csv")
  returns = to_daily_returns(prices)
  returns.to_csv("data/returns_daily.csv", index=False)
  PY
  ```

- **WRDS registry & hashes.** All loaders validate WRDS panels against `data/registry.json`. After refreshing CRSP data via your ingest pipeline (see `src/io/crsp_daily.py` for the canonical WRDS query), update the registry and commit the new hash:

  ```bash
  python tools/update_registry.py \
    --dataset data/returns_daily.csv \
    --wrds-source crsp.dsf \
    --note "CRSP daily returns refreshed from WRDS on $(date +%Y-%m-%d)"
  ```

  The command recomputes the SHA256, row counts, and date span before rewriting `data/registry.json`. `finance.io.load_returns_csv` aborts with a descriptive error if the on-disk file drifts from the recorded digest.

- **Week alignment & timezone.** Balanced panels assume Monday–Friday business weeks in America/New_York. The default `--drop-partial-weeks` policy removes short weeks; switch to `--impute-partial-weeks` if your source includes holidays or alternative sessions.
- **Run catalogues.** The latest tagged drops live in [`experiments/equity_panel/LATEST.md`](experiments/equity_panel/LATEST.md); browse local outputs with [`tools/list_runs.py`](tools/list_runs.py).

---

## 3. Repository map

| Path | Purpose |
| --- | --- |
| `src/finance`, `src/fjs`, `src/meta` | Core covariance, de-aliasing, and cache utilities. |
| `experiments/equity_panel/` | YAML configs coupled with the rolling runner. |
| `experiments/eval/` | Rolling daily evaluation (ΔMSE, VaR/ES, diagnostics). |
| `experiments/etf_panel/` | ETF sector/country demo built on the daily evaluation defaults. |
| `tools/` | CLI helpers (`clean_outputs`, `build_gallery`, `build_memo`, `build_brief`, `summarize_run`, etc.). |
| `src/report/` | Pandas/matplotlib helpers to assemble tables and plots. |
| `reports/templates/` | Jinja/Jinja-like templates for memos. |
| `tests/` | Unit/integration tests. Fixtures under `tests/report_fixtures/` drive reporting tests. |
| `figures/` | Auto-generated plots/tables (ignored by git). |
| `reports/` | Generated memos. |

---

## 4. Testing & linting

| Command | Scope |
| --- | --- |
| `make test-fast` | `pytest -m unit` (deterministic, small fixtures). |
| `make test-integration` | Smoke-level multi-module tests. |
| `make test-slow` | Long-running statistics/ablations. |
| `make test` | Entire pytest suite. |
| `make fmt` / `make lint` | Format via `black` + lint (`ruff`, `mypy`). |

Markers (`pytest.ini`):

- `unit` – default; always covered by CI.
- `integration` – smoke-style flows (cached IO, small data slices).
- `slow`, `heavy` – opt-in experiments; excluded from `make test-fast`.

---

## 5. Running equity experiments

### 5.1 Smoke slice (fast sanity check)

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py \
    --config experiments/equity_panel/config.smoke.yaml \
    --no-progress \
    --workers $(python -c 'import os;print(os.cpu_count() or 4)') \
    --assets-top 80 \
    --stride-windows 4 \
    --resume \
    --cache-dir .cache \
    --precompute-panel \
    --drop-partial-weeks \
    --estimator oas
```

Outputs land in `experiments/equity_panel/outputs_smoke/<design>_J*_solver-*_est-*_prep-*`.

### 5.2 Nested design

Add `--design nested --nested-replicates 5` to the command above or copy/adapt the nested configs under `experiments/equity_panel/`.

### 5.3 Crisis slices

Two crisis presets are provided:

- `experiments/equity_panel/config.crisis.2020.yaml`
- `experiments/equity_panel/config.crisis.2022.yaml`

Kick off a crisis run with the same CLI switches shown for the smoke slice, but substitute the crisis config and (optionally) tweak the estimator.

### 5.4 Release Candidate batch

`make rc` orchestrates:

1. Smoke runs for estimators `{dealias,lw,oas,cc,factor,tyler_shrink}` across the oneway design.
2. Nested smoke (`design=nested`) for a longer 2022 slice.
3. (If present) tagged crisis directories under `experiments/equity_panel/outputs_crisis_*`.
4. Gallery + memo generation (see sections 7 & 8).

`make rc-lite` is a spot-check pass that only runs `{dealias,lw,oas}` on the smoke and 2020 crisis configs before rebuilding the gallery and memo.

All RC targets respect the run policy flags (`--workers`, `--resume`, `--cache-dir .cache`, `--drop-partial-weeks`, etc.).

### 5.5 Robust MP edges

Edge detection margins can now be re-scaled with robust scatter estimates. The runner accepts `--edge-mode {scm,tyler,huber}`; the default `scm` leaves the historical behaviour untouched, while `tyler` and `huber` multiply the Marčenko–Pastur edge by a Tyler or Huber scatter ratio. Every window writes both values (`edge_scm`, `edge_selected`) to `detection_summary.csv` and the memo badges the chosen mode.

Example: compare stock SCM vs Tyler margins on the smoke slice.

```bash
PYTHONPATH=src python experiments/equity_panel/run.py \
    --config experiments/equity_panel/config.smoke.yaml \
    --no-progress --precompute-panel --drop-partial-weeks \
    --edge-mode tyler --estimator dealias
```

The same command with `--edge-mode scm` produces the benchmark run; the memo and `summary.json` include `"nested_scope": {"de_scoped_equity": true}` when nested windows are skipped solely by the `no_isolated_spike` guard.

### 5.6 Factor baselines

Two minimal factor models join the evaluation matrix:

- `--estimator factor_obs` plugs the observed-factor covariance (`Σ = BΣ_fBᵀ + Σ_ε`) via `src/evaluation/factor.py`. Supply factor returns with `--factor-csv path/to/factors.csv` (columns = factors).
- `--estimator poet` fits a POET-lite covariance with an automatic IC for `k` and diagonal residual shrinkage. No external factor file is required.

Both estimators appear in `metrics_summary.csv`, the aggregate tables, and the memo (look for the `Factor-Observed` and `POET-lite` rows). When factor data are missing the runner leaves a warning and skips the observed baseline gracefully.

### 5.7 Synthetic null/power harness

Run the null/power ROC harness to calibrate overlay gating:

```bash
make sweep:acceptance HARNESS_TRIALS=400
```

The target executes both `experiments/synthetic/null.py` and `experiments/synthetic/power.py`, persisting:

- score tables in `reports/synthetic/null_harness/` and `reports/synthetic/power_harness/`,
- ROC figures `reports/figures/roc_null.png` and `reports/figures/roc_power.png`,
- default thresholds + energy floor in `calibration_defaults.json`.

To tweak parameters manually:

```bash
PYTHONPATH=src python experiments/synthetic/null.py \
  --trials 600 --edge-modes scm tyler \
  --out reports/synthetic/null_harness --figures-out reports/figures

PYTHONPATH=src python experiments/synthetic/power.py \
  --trials 600 --mu-values 4 6 8 \
  --null-scores reports/synthetic/null_harness/null_scores.parquet \
  --out reports/synthetic/power_harness \
  --figures-out reports/figures \
  --defaults-path calibration_defaults.json
```

- `experiments/synthetic/calibrate_thresholds.py` now understands `--run-id`, `--resume`, and `--exec-mode {deterministic,throughput}`. Each cell persists to `reports/synthetic/calib/<RUN_ID>/cells/<cell>.json`, and every run writes `reports/synthetic/calib/<RUN_ID>/run.json` with git SHA, instance metadata, thread caps, and the chosen execution mode.
- Shard large sweeps with `tools/shard_grid.py --shards 4 --out manifest.jsonl ...`, then pass `--shard-manifest manifest.jsonl --shard-id k` (or set `SHARD_MANIFEST`/`SHARD_ID` when calling `make calibrate-thresholds`/`make aws:calibrate-thresholds`). Once all shards finish, consolidate via `tools/reduce_calibration.py --run-id <RUN_ID>` to regenerate `calibration/edge_delta_thresholds.json`, `calibration/defaults.json`, and ROC plots.
- The MP-edge routines now expose a configurable on-disk cache (default `.cache/mp_edges`). Override with `--mp-cache-dir /tmp/mp_cache` or disable via `--mp-cache-dir none` when comparing fresh runs.

### 5.8 Deterministic vs Throughput exec modes

- Every heavy runner (`experiments/synthetic/calibrate_thresholds.py`, `experiments/equity_panel/run.py`, and `experiments/eval/run.py`) accepts `--exec-mode {deterministic,throughput}`. Deterministic mode caps BLAS/OpenMP threads at 1 and uses the full worker count for reproducibility. Throughput mode allows 2–4 threads per worker (auto-selected based on `os.cpu_count()`) and scales the worker pool down accordingly.
- When dispatching via `scripts/aws_run.sh`, set `EXEC_MODE=throughput make aws:calibrate-thresholds ...` (or leave unset for deterministic). The wrapper exports the thread caps, records the mode inside `reports/runs/<RUN_ID>/run.json`, and forwards `EXEC_MODE` to the Make target so it can drive Python-side logic.
- Local `make calibrate-thresholds` calls honour `EXEC_MODE` as well. Combine with `MP_CACHE_DIR=...` or `RUN_ID=...` to keep the CLI invocations hermetic.

---

## 6. Calibration & Threshold Artefacts

### 6.1 Default acceptance profile

`calibration_defaults.json` (written by the sweep above) captures the working overlay configuration:

- `parameters`: `{delta, delta_frac, eps, stability_eta_deg, energy_floor, edge_mode}`
- `selection`: best-performing edge mode/energy floor with per-μ power and realised FPR
- `config`: simulation dimensions and seed for reproducibility

Reference this file from evaluation runs with `--gate-delta-calibration calibration_defaults.json`; the runner records the applied values in `detection_summary.csv` (`edge_*`, `delta_frac_used`, stability/leakage metrics).

### 6.2 Replicate-aware sweeps (optional)

The legacy replicate-aware sweep (`experiments/synthetic/calibrate_thresholds.py`) remains available when you need fine-grained delta/stability lookups per `(edge_mode, G, replicates_bin, p_bin)`. The helper script still works:

```bash
./scripts/run_calibration.sh          # auto-detects CPU count
./scripts/run_calibration.sh 32       # override worker count
```

or call the CLI directly (same flags as before). Commit updated JSON artefacts so WRDS jobs and smoke runs pick up the refreshed defaults.

### 5.8 Daily overlay diagnostics

Use `experiments/eval/run.py` for a lighter-weight daily pipeline that combines prewhitening, calm/crisis splits, and overlay diagnostics. It accepts wide (`date,<ticker>...`) or long (`date,ticker,ret`) returns with optional factor files and now supports capping the universe via `--assets-top`.

The rolling equity-panel experiment inherits the same observed-factor plumbing: pass `--prewhiten ff5mom --factor-csv data/factors/ff5mom_daily.csv` (or any registry entry) to `experiments/equity_panel/run.py` to regress out FF5+UMD exposure before nested balancing. The runner records the effective mode, factor columns, and mean R² in every `rolling_results.csv`, `summary.json`, and `run_meta.json`, and the memos/briefs surface those values under the new “Factor Baseline” section.

```bash
python experiments/eval/run.py \
    --returns-csv data/returns_daily.csv \
    --window 126 --horizon 21 \
    --assets-top 80 \
    --shrinker rie \
    --gate-delta-calibration calibration/edge_delta_thresholds.json \
    --gate-delta-frac-min 0.02 \
    --out reports/rc-YYYYMMDD/
```

Outputs per regime (`full`, `calm`, `crisis`) include:

- `metrics.csv`: ΔMSE vs shrinker baseline for EW/MV portfolios plus ES(95) errors.
- `risk.csv`: VaR/ES forecasts, realised tail means, and violation rates.
- `dm.csv`: Diebold–Mariano stats comparing the overlay against the shrinker baseline.
- `diagnostics.csv`: detection counts, edge margins, isolation share, and stability margins.
- `delta_mse.png`: bar plot of ΔMSE (created when matplotlib is available).

Flags of interest: `--factors-csv` to supply FF5+MOM data (falls back to an equal-weight MKT proxy), `--shrinker {rie,lw,oas,sample}` for non-detected directions, `--seed` to keep gating deterministic, and `--window`/`--horizon` to resize rolling windows.

### 5.9 Baseline coverage and gating defaults (Nov 2025)

- All daily runs now log shrinker parity for **sample, RIE, LW, OAS, CC, QuEST, EWMA, observed factor (FF5+MOM), and POET-lite** baselines. Failures are surfaced in `diagnostics.csv` under `baseline_error_*` columns.
- The strict overlay gate enforces calibrated δ-frac thresholds (see `calibration/edge_delta_thresholds.json`), minimum stability, admissible roots, and optional alignment guards. Use `--gate-mode soft` alongside `--gate-soft-max` for exploratory ranking.
- `--assets-top N` trims the alphabetically sorted universe before windowing, keeping bounded RC runs quick without re-sampling returns.

For an ETF demo (countries/sectors), run:

```bash
python experiments/etf_panel/run.py \
    --returns-csv data/etf_returns.csv \
    --out reports/etf-rc/
```

The ETF wrapper simply forwards options to the daily evaluation harness, emitting the same CSV/PNG diagnostics alongside a short overlay toggle note (`overlay_toggle.md`) that summarises when detections turn on or stay muted.

**Latest RC snapshot (4 Nov 2025)**

- Bounded DoW/Vol-state runs (126×21, top-80) land in `reports/rc-20251104/{dow-bounded,vol-bounded}/` with the usual metrics, risk, DM, diagnostics, and overlay toggle files.
- Memo and manifest: `reports/rc-20251104/memo.md`, `reports/rc-20251104/run_manifest.json`.
- Quick telemetry:

| Design | Regime | Detection rate | Substitution fraction | Median edge margin | Notes |
| --- | --- | --- | --- | --- | --- |
| DoW (RIE, FF5+MOM) | Full | 3.4 % | 5.6 % | 0.38 | Overlay beats RIE in calm bins; slight ΔMSE drag in crisis. |
| DoW (RIE, FF5+MOM) | Crisis | 4.6 % | 5.2 % | 0.41 | Gate held FPR surrogate ≤2 %; monitor late-2020 windows. |
| Vol-state (OAS, off) | Full | 2.9 % | 3.9 % | 0.33 | No prewhitening; overlay competitive with OAS in calm/mid. |
| Vol-state (OAS, off) | Crisis | 4.1 % | 3.5 % | 0.35 | Stable VaR95 coverage (±1%). |

Figures live under `figures/rc/20251104/` (generate via `make gallery`); rerun `make rc` after tuning to refresh both graphics and memo content.
| Smoke (oneway, 2023-01→03) | 4/4 windows (100%) | LW: −1.05×10⁻⁶, OAS: −1.11×10⁻⁶, CC: −7.5×10⁻⁸, Tyler: +2.57×10⁻¹ | Tyler vs De: p≈0.0137 (significant); every other DM test ≥0.074 | Shrinkage baselines dominate the aliased/de-aliased pair; see `figures/rc/oneway_J5_solver-auto_est-lw_prep-none/plots/dm_pvals.png` and `.../edge_margin_hist.png`. |
| Nested smoke (2022-01→2023-12) | 0/24 windows (0%) | ΔMSE columns remain ≈0 because every window is skipped by guardrails | DM stats effectively degenerate | Memo now badges the run with “no accepted detections; check guardrails”; see `figures/rc/nested_J5_solver-auto_est-dealias_prep-none/tables/estimators.csv` plus `summary.json`’s `nested_skip_reasons`. |
| Crisis 2020 (oneway, 2020-02-15→05-31) | 4/4 windows (100%) | De-aliased median ΔMSE vs LW: +2.18×10⁻⁵ (worse); vs Tyler: +0.12 | DM(p) vs LW ≈ 1.1×10⁻⁴, vs OAS ≈ 9.3×10⁻⁵, vs Tyler ≈ 9.5×10⁻⁴ | De-aliased loses to shrinkage baselines but detections are plentiful; browse `figures/rc/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531/plots/*.png`. |

Memo extras:

- **Key Results panel:** compact estimator-by-run table with detection, ΔMSE, CIs, DM p-values, edge margins, and window counts.
- **Rejection summary:** populated once guardrails trigger; the crisis slice still shows only the “other” bookkeeping bucket.
- **Ablation snapshot:** automatically embeds when `ablation_summary.csv` exists; the present drop shows a placeholder because the new smoke ablation grid timed out before completing.

Artifacts of interest:

- Tables/plots per run: `figures/rc/<run_tag>/tables/*.csv|md|tex`, `figures/rc/<run_tag>/plots/*.png`.
- Memo Markdown: `reports/memo.md` and timestamped copies under `reports/`.
- Advisor brief: `reports/brief.md` plus timestamped copies under `reports/`.
- Summary diagnostics (edge hist, isolation bars, stability scatter): `figures/rc/summary/*.png`.
- Crisis CSVs/plots: `experiments/equity_panel/outputs_crisis_2020/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531` (similar tags appear for other crises).

**Action items before the next RC**

- Fix nested detection (0 % coverage) so the memo badge can be retired.
- Investigate crisis-step tuning: the 2020 slice shows de-aliased ≫ shrinkage MSE and very small edge buffers despite perfect coverage.
- Either relax runtime limits or shrink the grid further so `experiments/equity_panel/config.ablation.smoke.yaml` emits `ablation_summary.csv` and the gallery plots `ablation_heatmap.png`.

---

## 6. Smoke hygiene & summarisation

- Archive or purge untagged outputs:
  ```bash
  python tools/clean_outputs.py --dry-run   # preview moves
  python tools/clean_outputs.py --purge     # clear legacy root files
  ```

  Tagged directories (`<design>_J*_solver-*_est-*_prep-*`) are preserved; legacy assets move to `archived/` unless `--purge` is supplied.

- Text summary (auto-detects tagged directories):
  ```bash
  python tools/summarize_run.py experiments/equity_panel/outputs_smoke
  ```

  The script checks for tagged children first, warns when it skips legacy outputs, prints detection coverage, ΔMSE, DM stats, and points to gallery artifacts if they exist.

---

## 7. Gallery generation

### 7.1 Smoke gallery

```bash
make gallery
```

Reads `experiments/equity_panel/config.gallery.yaml`, collects matching runs (prefers tagged directories), and writes:

- `figures/<gallery_name>/<run_tag>/tables/estimators.{csv,md,tex}`
- `figures/<gallery_name>/<run_tag>/tables/rejections.{csv,md,tex}`
- `figures/<gallery_name>/<run_tag>/plots/`
  - `dm_pvals.png` (grouped DM p-values, EW/MV)
  - `detection_rate.png`
  - `edge_margin_hist.png`
  - Optional `ablation_heatmap.png` when `ablation_summary.csv` is present.

### 7.2 RC gallery

`make rc` automatically runs `tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml`, writing to `figures/rc/`. The YAML lists smoke, nested, and crisis directories; update it to add new estimators or additional crisis runs.

Gallaries use the same reporting helpers (`src/report/gather.py`, `src/report/tables.py`, `src/report/plots.py`). Tagged runs guarantee consistent naming and ensure the metrics map cleanly across releases.

---

## 8. Memo workflow

`tools/build_memo.py` consumes the same YAML used for gallery generation and renders a Markdown memo using `reports/templates/memo.md.j2`. The memo contains:

- Short problem statement.
- Run selection summary (design, replicates, date window, estimators).
- Key estimator table (detection rate, ΔMSE, DM p-values).
- Bullets on detection coverage, ΔMSE directionality, and DM significance.
- Reason-code summary table with dominant gating outcomes.
- Diagnostics snapshot (edge-margin histogram, isolation-share bars, direction–stability scatter).
- Links to tables/plots and the per-run `run_meta.json`.

Outputs:

- `reports/memo.md`
- `reports/memo_<timestamp>.md`

CI (`.github/workflows/smoke.yml`) runs the gallery + memo steps for smoke runs and uploads `figures/` plus `reports/memo.md` as artifacts.

---

### 8.1 Advisor brief

`tools/build_brief.py` renders a single-page summary targeting advisors. It reuses the gallery YAML and surfaces:

- Detection coverage and DM significance highlights.
- Top gating reason with recommended follow-up.
- A compact reason-code table for each run.

Outputs:

- `reports/brief.md`
- `reports/brief_<timestamp>.md`

Include the brief alongside the memo when circulating RC updates.

---

## 8. Estimators & preprocessing

| CLI flag | Description |
| --- | --- |
| `--estimator {aliased,dealias,lw,oas,cc,factor,tyler_shrink}` | Selects the covariance estimator used for the forecast and cache key. |
| `--factor-csv path/to/factors.csv` | Enables the observed-factor covariance (OLS fit with intercept). |
| `--winsorize q` | Clip daily returns column-wise to `[q, 1−q]` quantiles before balancing. |
| `--huber c` | Huber clip (median ± c·MAD). Mutually exclusive with `--winsorize`. |
| `--estimator tyler_shrink` | Tyler M-estimator with ridge regularisation (see `src/finance/robust.py`). |

Preprocess selections propagate into the cache key, panel manifest, artifact directories, run metadata, and the tables/plots rendered from gallery builds.

---

## 9. Calibration & gating

- **q-discipline gate.** All equity configs inherit a `gating` block (defaults below). Set `enable: false` to revert to legacy behaviour, tune `q_max` to cap accepted detections per window, and keep `require_isolated: true` to avoid substituting when no MP-isolated spike exists.

  ```yaml
  gating:
    enable: true
    q_max: 2
    require_isolated: true
    mode: fixed
    calibration_path: calibration/edge_delta_thresholds.json
  ```

- **Per-window diagnostics.** `detection_summary.csv` now records `skip_reason`, `isolated_spikes`, `gate_discarded_count`, a JSON payload of discarded detections, and the alignment statistics `angle_min_deg` / `energy_mu` (principal angle and quadratic form against Σ̂).
- **Observed coverage (Oct 2025 calibration).** With the defaults below, the smoke slice substitutes 3/4 windows (one skip tagged `no_isolated_spike`, median alignment ≈17°) and the 2020 crisis slice substitutes all 6 windows (median alignment ≈31°). No detections were clipped by the top‑K gate, so `gate_discarded` stayed empty.

- **Summary telemetry.** `summary.json` includes a `gating` section with substitution counts and skip reasons, plus `nested_skip_details` (years kept, common ISO weeks, replicates, exit reason) whenever nested prep fails to assemble a valid balanced block.
- **Calibrated gating workflow (RC).**
  1. `PYTHONPATH=src python experiments/synthetic/power_null.py --design oneway --edge-modes scm tyler --calibrate-delta --alpha 0.01 --out calibration/edge_delta_thresholds.json`
  2. `PYTHONPATH=src python experiments/equity_panel/run.py --config … --gating-mode calibrated --gating-calibration calibration/edge_delta_thresholds.json …`
  3. `python tools/build_gallery.py …`, `python tools/build_memo.py …`, `python tools/aggregate_runs.py …`

  The runner logs `gating_mode`, `delta_frac_used` per window, and calibration misses in `summary.json`, `detection_summary.csv`, and `metrics_summary.csv` so the memo can badge edge/gate combinations.

- **Acceptance sweep.** `experiments/equity_panel/sweep_acceptance.py` reuses the runner plumbing to scan acceptance parameters. For example:

  ```bash
  PYTHONPATH=src python experiments/equity_panel/sweep_acceptance.py \
    --config experiments/equity_panel/config.smoke.yaml \
    --design oneway \
    --grid default \
    --estimators dealias lw oas cc tyler
  ```

  The script spins up tagged sub-runs and emits `sweep_summary.csv` with detection rate, median edge margin, substitution share, and DM(MSE/QLIKE) deltas versus LW/OAS.
  Two overview heatmaps—`E5_detection_rate.png` and `E5_mse_gain.png`—summarise the detection density and the (Aliased − De-aliased) ΔMSE surface.

- **Alignment plots.** `tools/build_gallery.py` adds `alignment_angles.png` when detections contain alignment diagnostics, complementing the existing DM/detection/edge plots.

| Key | RC default | Meaning |
| --- | --- | --- |
| `dealias_delta_frac` | `0.02` | Relative MP edge buffer (dominant guardrail). |
| `dealias_eps` | `0.03` | Minimum t-vector mass on the target component. |
| `stability_eta_deg` | `0.4` | Angular jitter for stability verification (kept to preserve the historical guardrail). |
| `a_grid` | `120` | Angular resolution for the t-vector search; balances runtime vs. alignment smoothness. |
| `gating.q_max` | `2` | Maximum detections substituted per window (top‑K selection score = energy × stability). |
| `edge_mode` | `scm` | Edge estimator driving the MP threshold (`tyler` / `huber` supported per run). |
| `gating.mode` | `fixed` | `fixed` uses the config `delta_frac`; `calibrated` takes `max(config, lookup)` from `calibration/edge_delta_thresholds.json`. |
| `gating.calibration_path` | `calibration/edge_delta_thresholds.json` | Mapping `(p,T) → delta_frac_min` generated by the synthetic null harness. |
| `alignment_top_p` | `3` | PCA dimensionality used in the alignment diagnostic. |
| `off_component_leak_cap` | `10.0` | Reject when Σ₂ leakage exceeds the cap relative to Σ₁. |
| `energy_min_abs` | `1e-6` | Drop spikes with insufficient Σ₁ energy. |
| `sigma_ablation` | `False` | Toggle ±10% Cs perturbations for sensitivity checks. |

All `make rc-*` targets honour `RC_Q_MAX` and `RC_GATE_DELTA_FRAC_MIN` (defaults `2` and `0.01` for the exploratory `rc-lite` path). Override them on the command line—e.g., `RC_Q_MAX=1 RC_GATE_DELTA_FRAC_MIN=0.02 make rc-dow`—when you need the historical guardrails.

The detection summary and memo bullets explicitly badge runs where `no_isolated_spike` skips dominate; leverage the sweep tool above to retune acceptance thresholds when that happens. The October 2025 sweep over `(δ_frac, ε, η, a_grid) ∈ {0.01,0.02} × {0.02,0.03} × {0.4,0.6} × {90,120}` produced numerically identical detection and ΔMSE profiles across the grid, so the RC defaults above retain the long-standing guardrail (δ_frac = 0.02, ε = 0.03, η = 0.4) while standardising on `a_grid = 120` to keep the alignment diagnostic smooth without materially increasing runtime. The refreshed `metrics_summary.csv` mirrors these decisions via `edge_mode`, `gating_mode`, `substitution_fraction`, `skip_no_isolated_share`, and VaR/ES p-value columns that propagate into the gallery tables and aggregates.

---

## 10. Aggregating runs

Combine multiple `metrics_summary.csv` files without rebuilding the memo. A typical RC batch stitches the SCM smoke baseline, Tyler/Hüber crisis slices (fixed gate), the calibrated Tyler reruns, and the nested slice:

```bash
python3 tools/aggregate_runs.py --inputs \
  experiments/equity_panel/outputs_smoke_scm/oneway_J5_solver-auto_est-dealias_prep-none \
  experiments/equity_panel/outputs_crisis_tyler/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 \
  experiments/equity_panel/outputs_smoke_huber/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 \
  experiments/equity_panel/outputs_smoke_calibrated/oneway_J5_solver-auto_est-dealias_prep-none \
  experiments/equity_panel/outputs_smoke_calibrated/oneway_J5_solver-auto_est-dealias_prep-none__crisis_20200215_20200531 \
  experiments/equity_panel/outputs_nested_smoke/nested_J5_solver-auto_est-dealias_prep-none \
  --out reports/aggregate_summary.csv
```

Every aggregated row now carries `edge_mode`, `gating_mode`, `substitution_fraction`, `skip_no_isolated_share`, `delta_frac_used_min/max`, and VaR diagnostics (`var_kupiec_p`, `var_independence_p`, `es_shortfall_p`) sourced from each run’s `metrics_summary.csv`. Add `--tex-out <path>` if you also need a LaTeX table.

---

## 11. CI overview

`.github/workflows/smoke.yml` executes:

1. `make test-fast`
2. Smoke equity run (OAS estimator, cached).
3. `tools/build_gallery.py --config experiments/equity_panel/config.gallery.yaml`
4. `tools/build_memo.py --config experiments/equity_panel/config.gallery.yaml`
5. `tools/summarize_run.py experiments/equity_panel/outputs_smoke`
6. Uploads `figures/` and `reports/memo.md` as artifacts.

This ensures smoke regressions surface quickly and reviewers always have current plots/tables.

---

## 12. Tips & troubleshooting

- **Cache hygiene**: Use `--resume --cache-dir .cache` to keep per-window statistics; delete `.cache/` when changing estimator/preprocess combos that affect the cache key signature.
- **Tagged directories**: Prefer the `<design>_J*_solver-*_est-*_prep-*` naming convention. `tools/clean_outputs.py` moves legacy files aside and `tools/summarize_run.py` warns when legacy outputs are ignored.
- **Adding estimators**: Extend `src/finance/eval.py`, `src/report/gather.py`, and update the DM suffix list (`DM_SUFFIXES`) to expose new pairwise statistics. Update both `config.gallery.yaml` and `config.rc.yaml` to include the new estimator in gallery/memo builds.
- **Extending memos**: Modify `reports/templates/memo.md.j2` or pass additional context from `tools/build_memo.py` (e.g., more bullets, external benchmarks).
- **Local figure cleanup**: Remove stale galleries with `rm -rf figures/<name>` and regenerate via `make gallery`/`make rc`.

---

Happy modelling! Run `make rc` before major reviews to ensure tables, plots, and memos reflect the latest code and configurations.
