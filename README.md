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

## 2. Repository map

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

## 3. Testing & linting

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

## 4. Running equity experiments

### 4.1 Smoke slice (fast sanity check)

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

### 4.2 Nested design

Add `--design nested --nested-replicates 5` to the command above or copy/adapt the nested configs under `experiments/equity_panel/`.

### 4.3 Crisis slices

Two crisis presets are provided:

- `experiments/equity_panel/config.crisis.2020.yaml`
- `experiments/equity_panel/config.crisis.2022.yaml`

Kick off a crisis run with the same CLI switches shown for the smoke slice, but substitute the crisis config and (optionally) tweak the estimator.

### 4.4 Release Candidate batch

`make rc` orchestrates:

1. Smoke runs for estimators `{dealias,lw,oas,cc,factor,tyler_shrink}`.
2. Nested smoke (`design=nested`).
3. Crisis windows (`config.crisis.2020.yaml`, `config.crisis.2022.yaml`) if the outputs directory exists.
4. Gallery + memo generation (see sections 5 & 6).

All runs respect the smoke-sized window (`6+1` weeks) and the caching flags defined by the run policy.

**Latest RC snapshot (28 Oct 2025)**

| Regime | Detection rate | ΔMSE (EW) vs LW | ΔMSE (EW) vs OAS | DM highlights | Commentary |
| --- | --- | --- | --- | --- | --- |
| Smoke (oneway) | 4/4 windows (100%) | −3.6×10⁻⁷ | −3.7×10⁻⁷ | De vs Tyler: p=0.074 (non-sig) | De-aliased beats Aliased/SCM but shrinkage (LW/OAS) still halves MSE; Tyler M-estimator is unstable (ΔMSE≈+0.26). |
| Nested (Year⊃Week) | 0/24 windows (0%) | n/a | n/a | n/a | Guardrails reject all candidates; consider relaxing `dealias_delta_frac` or lengthening the window if nested signal is needed. |
| Crisis 2020 (Feb–May) | 5/5 windows (100%) | −2.0×10⁻⁵ | −2.1×10⁻⁵ | DM p≈6×10⁻⁵ (stat≈17) vs LW | De-aliased forecasts are materially worse than LW/OAS during the COVID drawdown; shrinkage dominates. |
| Crisis 2022 (Sep–Nov) | 3/3 windows (100%) | −1.0×10⁻⁶ | −1.1×10⁻⁶ | De vs Tyler: p=0.014 (Tyler degraded) | Shrinkage again outperforms; de-aliased lifts variance modestly. |

Artifacts:

- RC tables/plots: `figures/rc/<run_tag>/{tables,plots}/` (e.g. `figures/rc/oneway_J5_solver-auto_est-oas_prep-none/plots/dm_pvals.png`).
- Crisis tables mirror the tagged directories under `experiments/equity_panel/outputs_crisis_{2020,2022}/`.
- Memo digest: `reports/memo.md` (updated by `make rc` and in CI artifacts).

**Action items:** benchmark adjustments (winsorize/huber, alternative guardrails) are needed before relying on de-aliased forecasts in crisis regimes; shrinkage remains the safest baseline on the current smoke slice.

---

## 5. Smoke hygiene & summarisation

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

## 6. Gallery generation

### 6.1 Smoke gallery

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

### 6.2 RC gallery

`make rc` automatically runs `tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml`, writing to `figures/rc/`. The YAML lists smoke, nested, and crisis directories; update it to add new estimators or additional crisis runs.

Gallaries use the same reporting helpers (`src/report/gather.py`, `src/report/tables.py`, `src/report/plots.py`). Tagged runs guarantee consistent naming and ensure the metrics map cleanly across releases.

---

## 7. Memo workflow

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

## 9. Detection guardrails (key config entries)

| Key | Default | Meaning |
| --- | --- | --- |
| `dealias_delta_frac` | `0.02` | Relative MP edge buffer (dominant guardrail). |
| `dealias_eps` | `0.03` | Minimum t-vector mass on the target component. |
| `stability_eta_deg` | `0.4` | Angular jitter for stability verification. |
| `off_component_leak_cap` | `10.0` | Reject when Σ₂ leakage exceeds the cap relative to Σ₁. |
| `energy_min_abs` | `1e-6` | Drop spikes with insufficient Σ₁ energy. |
| `sigma_ablation` | `False` | Toggle ±10% Cs perturbations for sensitivity checks. |

Detection summaries (`detection_summary.csv`) and aggregated `summary.json` provide per-window and overall guardrail diagnostics (edge margins, rejection counts, leak ratios).

---

## 10. CI overview

`.github/workflows/smoke.yml` executes:

1. `make test-fast`
2. Smoke equity run (OAS estimator, cached).
3. `tools/build_gallery.py --config experiments/equity_panel/config.gallery.yaml`
4. `tools/build_memo.py --config experiments/equity_panel/config.gallery.yaml`
5. `tools/summarize_run.py experiments/equity_panel/outputs_smoke`
6. Uploads `figures/` and `reports/memo.md` as artifacts.

This ensures smoke regressions surface quickly and reviewers always have current plots/tables.

---

## 11. Tips & troubleshooting

- **Cache hygiene**: Use `--resume --cache-dir .cache` to keep per-window statistics; delete `.cache/` when changing estimator/preprocess combos that affect the cache key signature.
- **Tagged directories**: Prefer the `<design>_J*_solver-*_est-*_prep-*` naming convention. `tools/clean_outputs.py` moves legacy files aside and `tools/summarize_run.py` warns when legacy outputs are ignored.
- **Adding estimators**: Extend `src/finance/eval.py`, `src/report/gather.py`, and update the DM suffix list (`DM_SUFFIXES`) to expose new pairwise statistics. Update both `config.gallery.yaml` and `config.rc.yaml` to include the new estimator in gallery/memo builds.
- **Extending memos**: Modify `reports/templates/memo.md.j2` or pass additional context from `tools/build_memo.py` (e.g., more bullets, external benchmarks).
- **Local figure cleanup**: Remove stale galleries with `rm -rf figures/<name>` and regenerate via `make gallery`/`make rc`.

---

Happy modelling! Run `make rc` before major reviews to ensure tables, plots, and memos reflect the latest code and configurations.
