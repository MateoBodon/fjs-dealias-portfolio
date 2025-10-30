# fjs-dealias-portfolio

Robust variance forecasting over balanced equity panels, with tooling to explore alternative estimators, crisis windows, and reporting assets. The project contains the de-aliased (FJS) covariance pipeline, supporting shrinkage baselines, preprocess toggles, and an automated gallery/memo workflow for review.

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

- **Week alignment & timezone.** Balanced panels assume Monday–Friday business weeks in America/New_York. The default `--drop-partial-weeks` policy removes short weeks; switch to `--impute-partial-weeks` if your source includes holidays or alternative sessions.
- **Run catalogues.** The latest tagged drops live in [`experiments/equity_panel/LATEST.md`](experiments/equity_panel/LATEST.md); browse local outputs with [`tools/list_runs.py`](tools/list_runs.py).

---

## 3. Repository map

| Path | Purpose |
| --- | --- |
| `src/finance`, `src/fjs`, `src/meta` | Core covariance, de-aliasing, and cache utilities. |
| `experiments/equity_panel/` | YAML configs coupled with the rolling runner. |
| `tools/` | CLI helpers (`clean_outputs`, `build_gallery`, `build_memo`, `summarize_run`, etc.). |
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

**Latest RC snapshot (30 Oct 2025)**

The gallery for this drop lives under `figures/rc/`, and the memo digest is in `reports/memo.md` (rendered via the new key-results panel, rejection summary, and ablation placeholder). Highlights:

| Regime | Detection rate | ΔMSE (EW) vs De-aliased | DM highlights | Commentary & figures |
| --- | --- | --- | --- | --- |
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
- Links to tables/plots and the per-run `run_meta.json`.

Outputs:

- `reports/memo.md`
- `reports/memo_<timestamp>.md`

CI (`.github/workflows/smoke.yml`) runs the gallery + memo steps for smoke runs and uploads `figures/` plus `reports/memo.md` as artifacts.

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
  ```

- **Per-window diagnostics.** `detection_summary.csv` now records `skip_reason`, `isolated_spikes`, `gate_discarded_count`, a JSON payload of discarded detections, and the alignment statistics `angle_min_deg` / `energy_mu` (principal angle and quadratic form against Σ̂).
- **Observed coverage (Oct 2025 calibration).** With the defaults below, the smoke slice substitutes 3/4 windows (one skip tagged `no_isolated_spike`, median alignment ≈17°) and the 2020 crisis slice substitutes all 6 windows (median alignment ≈31°). No detections were clipped by the top‑K gate, so `gate_discarded` stayed empty.

- **Summary telemetry.** `summary.json` includes a `gating` section with substitution counts and skip reasons, plus `nested_skip_details` (years kept, common ISO weeks, replicates, exit reason) whenever nested prep fails to assemble a valid balanced block.

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
| `alignment_top_p` | `3` | PCA dimensionality used in the alignment diagnostic. |
| `off_component_leak_cap` | `10.0` | Reject when Σ₂ leakage exceeds the cap relative to Σ₁. |
| `energy_min_abs` | `1e-6` | Drop spikes with insufficient Σ₁ energy. |
| `sigma_ablation` | `False` | Toggle ±10% Cs perturbations for sensitivity checks. |

The detection summary and memo bullets explicitly badge runs where `no_isolated_spike` skips dominate; leverage the sweep tool above to retune acceptance thresholds when that happens. The October 2025 sweep over `(δ_frac, ε, η, a_grid) ∈ {0.01,0.02} × {0.02,0.03} × {0.4,0.6} × {90,120}` produced numerically identical detection and ΔMSE profiles across the grid, so the RC defaults above retain the long-standing guardrail (δ_frac = 0.02, ε = 0.03, η = 0.4) while standardising on `a_grid = 120` to keep the alignment diagnostic smooth without materially increasing runtime.

---

## 10. Aggregating runs

Combine multiple `metrics_summary.csv` files without rebuilding the memo:

```bash
python tools/aggregate_runs.py \
  --inputs "experiments/equity_panel/outputs_smoke/*" \
  --out reports/aggregate_summary.csv \
  --tex-out reports/aggregate_summary.tex
```

Each row is tagged with `run`, `run_path`, `crisis_label`, and `design` (sourced from `summary.json`), and an optional LaTeX table is emitted when `--tex-out` is supplied.

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
