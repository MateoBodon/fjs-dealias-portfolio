# AGENTS.md — fjs-dealias-portfolio

## Primary Goals

- Study FJS-style MANOVA de-aliasing as an overlay on covariance estimators for equity-return panels (DoW, nested, vol-state designs).
- Calibrate acceptance thresholds under MP null/power and evaluate performance vs strong shrinkage and factor baselines.
- Produce reproducible RC drops (`reports/rc-YYYYMMDD/`) with memos/briefs that are safe to hand to Prof. Fan.

---

## Environment & Setup

- Language: Python 3.11+.
- Install (dev mode):

  - `pip install -e .[dev]`
  - `pre-commit install` (if `.pre-commit-config.yaml` exists).

- Helpful env vars:
  - `EXEC_MODE={deterministic,throughput}` — execution mode for heavy runners; use `throughput` on Hetzner.:contentReference[oaicite:37]{index=37}  
  - `MP_CACHE_DIR` — cache directory for MP edge computations when comparing runs.:contentReference[oaicite:38]{index=38}  

---

## Data & Secrets

- WRDS data is **never** committed. Only commit:
  - Panels derived into `data/returns_daily.csv` / factor CSVs.
  - Registry metadata in `data/registry.json`.:contentReference[oaicite:39]{index=39}  

- WRDS connection details live outside the repo (e.g. `.pgpass`, env vars); **do not** print credentials or full SQL queries into logs.

- Use `tools/update_registry.py` to update `data/registry.json` after refreshing WRDS exports. It recomputes hashes and row-counts and will cause loaders to fail fast if the file drifts.:contentReference[oaicite:40]{index=40}  

---

## Typical Commands

### Testing & Linting

- Fast unit tests: `make test-fast`
- Integration tests: `make test-integration`
- Full test suite: `make test`
- Slow/ablations (opt-in): `make test-slow`
- Format & lint: `make fmt && make lint`:contentReference[oaicite:41]{index=41}  

### Equity Panel Experiments

- Smoke slice (local sanity, small universe):​:contentReference[oaicite:42]{index=42}  

  ```bash
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py \
      --config experiments/equity_panel/config.smoke.yaml \
      --no-progress \
      --workers "$(python -c 'import os; print(os.cpu_count() or 4)')" \
      --assets-top 80 \
      --stride-windows 4 \
      --resume \
      --cache-dir .cache \
      --precompute-panel \
      --drop-partial-weeks \
      --estimator oas
Nested design: add --design nested --nested-replicates 5 or use the nested configs under experiments/equity_panel/.
GitHub
Crisis slices (2020, 2022): use config.crisis.2020.yaml / config.crisis.2022.yaml with the same CLI skeleton.
GitHub
Release Candidate batch (heavy; run on Hetzner):
make rc — full RC (smoke + nested + crises + gallery + memo/brief).
make rc-lite — quick RC-lite for {dealias,lw,oas} only.
make rc-lite-sanity — deterministic DoW/Vol sanity check with telemetry in reports/rc-<DATE>/summary_sanity.json.
GitHub
+1
Synthetic Calibration
Null/power ROC sweep:
make sweep:acceptance HARNESS_TRIALS=400
Targeted tweaks:
bash
Copy code
PYTHONPATH=src python experiments/synthetic/null.py \
    --trials 600 --edge-modes scm tyler \
    --out reports/synthetic/null_harness --figures-out reports/figures

PYTHONPATH=src python experiments/synthetic/power.py \
    --trials 600 --mu-values 4 6 8 \
    --null-scores reports/synthetic/null_harness/null_scores.parquet \
    --out reports/synthetic/power_harness \
    --figures-out reports/figures \
    --defaults-path calibration_defaults.json
Then use tools/reduce_calibration.py to refresh calibration/edge_delta_thresholds.json.
GitHub
Daily Evaluation & Overlay Diagnostics
Daily overlay run (fast-ish):
bash
Copy code
PYTHONPATH=src python experiments/eval/run.py \
    --returns-csv data/returns_daily.csv \
    --window 126 --horizon 21 \
    --assets-top 80 \
    --shrinker rie \
    --gate-delta-calibration calibration/edge_delta_thresholds.json \
    --gate-delta-frac-min 0.02 \
    --out reports/rc-YYYYMMDD/
This writes metrics.csv, risk.csv, dm.csv, diagnostics.csv, and plots like delta_mse.png and flip_dm.png.
GitHub
+1
Coding & Style Guidelines
Python:
Type hints on public functions.
Keep functions short and composable; avoid clever one-liners in core logic.
Use apply_patch-style edits (Codex knows what that means).
Numerical / stats:
Prefer vectorized operations and stable numerics.
Avoid silent changes to calibration defaults; if a behavior change is intentional, update calibration_defaults.json or the relevant config + memo.
Testing Expectations
When you change anything in src/, experiments/, or tools/:
Run make test-fast.
If your change affects runners/calibration, also run:
make rc-lite-sanity, and
the smallest relevant synthetic harness (make sweep:acceptance with reduced trials or a single null/power call).
For nested or crisis changes, re-run the smallest nested/cisis configs and check:
detection coverage,
ΔMSE vs shrinkers,
DM p-values on the flip set.
GitHub
Log heavy test output to reports/ or figures/rc/ as appropriate; don’t spam stdout with massive tables.
Docs, RCs & Progress Logging
Use tools/build_gallery.py, tools/build_memo.py, and tools/build_brief.py to generate advisor-facing RC artifacts.
GitHub
+1
Every time you run an RC or important calibration:
Append a bullet to PROGRESS.md with:
Date, git SHA, machine (local/Hetzner), key configs.
High-level metrics (detection %, ΔMSE, DM p-values, coverage).
Paths to metrics_summary.csv, memo, and key figures.
Git & Branching
Always work on feature branches, preferably codex/<short-task> for Codex-driven work.
Commit messages:
feat:, fix:, refactor:, test:, docs:, perf: prefixes.
Short, imperative: feat: add nested gating diagnostics.
For Codex:
Do not revert unrelated user changes. If you see unexpected diffs, stop and report.
Prefer small, reviewable commits over massive ones.
Hetzner & Heavy Jobs
For heavy calibrations and full RC batches, assume Codex is running on the Hetzner box:
Use --profile fjs-hetzner (see config.toml).
Prefer EXEC_MODE=throughput for long sweeps.
Keep OMP_NUM_THREADS and worker counts reasonable; don’t overload the box.
If you’re on a local laptop profile, restrict yourself to:
make test-fast,
smoke equity/eval runs,
small synthetic experiments.
What You Must Do Before Declaring a Task “Done”
All relevant tests green (make test-fast at minimum).
If behavior changed:
Update config files and/or calibration JSONs.
Update docs (README, PLAN.md, memo templates) as needed.
Append an entry to PROGRESS.md.
Commit and, if configured, push your branch.
In your final message, summarize:
What changed.
Which commands you ran.
Where to find artifacts.