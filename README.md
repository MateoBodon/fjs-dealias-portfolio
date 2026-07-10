# fjs-dealias-portfolio

Robust covariance forecasting and portfolio gating over balanced equity panels, with synthetic calibration harnesses, weekly/daily evaluation runners, and reproducible reporting.

---

## Current Status (2026-07-03)

- **AI OS v2 installed:** current strategy docs live in `docs/strategy/`, state-audit docs live in `project_state/`, and pre-v2 docs are indexed under `docs/_archive/pre_ai_os_v2/20260703/`.
- **Engineering baseline:** auditability and reproducibility are strong (`docs/agent_runs/`, `PROGRESS.md`, bundle/test/runlog gates).
- **Minimum commit gate:** `. .venv/bin/activate && make test-fast`.
- **Current recovered high-water mark:** T-012 daily DoW four-leg matrix is recovered and scientifically useful, but not cleanly ratified because monitoring/audit preservation failed.
- **Latest uncapped daily evidence:** `reports/rc-ticket-07-20251222_183800/summary/` (`cap_active=false`, `window_coverage=1.0`, `n_effective_mse=1749` in `summary_perf.csv`).
- **Main research blocker:** injection sensitivity for week design is still flat-zero in `reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv` (detection and acceptance both 0.0 for `mu` in {0, 3, 6, 12, 24}).
- **Top priorities:**
  1. Debug flat-zero injection response (or conclusively explain the theory/data mismatch).
  2. Produce one advisor-ready uncapped run with valid aligned comparisons and clear effect reporting.
- **Scope rule:** do not expand experiment grids until those two gates are closed.

---

## Quick start

```bash
git clone https://github.com/MateoBodon/fjs-dealias-portfolio.git
cd fjs-dealias-portfolio
python -m venv .venv
source .venv/bin/activate
make setup          # installs editable + dev deps
make test-fast      # unit suite (required before commits)
```

### Data (required hashes)
- `data/returns_daily.csv` — sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`
- `data/factors/ff5mom_daily.csv` — sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`
Registries: `data/registry.json`, `data/factors/registry.json`.

### Determinism & exec mode
Set `EXEC_MODE=deterministic` for thread-capped runs (used for tests, smokes, calibrations). Throughput mode is available but not the default.

---

## Key commands

| Purpose | Command | Outputs |
| --- | --- | --- |
| Unit tests | `make test-fast` | - |
| Project State Audit Bundle | `make project-state-audit-bundle` | `reports/_bundles/<stamp>_repo_project-state_initial.zip` |
| AI OS review bundle | `make ai-os-review-bundle RUN_LOG=<path> [STATE_BUNDLE=<zip>]` | `reports/_bundles/<stamp>_repo_review_<ticket>.zip` |
| Tiny nested smoke (deterministic, capped windows=3) | `make run:equity_nested_smoke_tiny` | `experiments/equity_panel/outputs_nested_smoke_tiny/` |
| RC-lite spot-check (deterministic) | `make rc-lite` | `reports/rc-<DATE>/` + gallery/memo |
| Null/power sweep (overlay calibration) | `HARNESS_TRIALS=800 EXEC_MODE=deterministic make sweep:acceptance` | `reports/synthetic/{null_harness,power_harness}`, `calibration_defaults.json` |
| GPT review bundle | `make gpt-bundle TICKET=<ticket> RUN_NAME=<run_name>` | `docs/gpt_bundles/<stamp>_<ticket>_<run_name>.zip` |

---

## Runners & configs

- **Weekly equity panel:** `experiments/equity_panel/run.py`
  - Configs: `experiments/equity_panel/config*.yaml` (includes nested smoke/crisis).
  - CLI highlights: `--design {oneway,nested,dow,vol}`, `--nested-replicates`, `--edge-mode {scm,tyler,huber}`, `--gating-mode {fixed,calibrated}`, `--gating-calibration path`, `--max-windows` (caps evaluated windows; persisted in `config_resolved.yaml`), `--prewhiten {off,ff5,ff5mom,custom}`, `--factor-csv`, `--exec-mode {deterministic,throughput}`.
  - Make targets: `run:equity_smoke`, `run:equity_nested_smoke_tiny`, `rc`, `rc-lite`, `rc-lite-sanity`.

- **Daily eval:** `experiments/eval/run.py` (ΔMSE/QLIKE/coverage; crisis slices; supports `--max-windows`, factor prewhiten, gating calib).

- **Synthetic calibration:** `experiments/synthetic/calibrate_thresholds.py`, `null.py`, `power.py`, `nested_killtest.py` (design-aware gating; defaults point at nested calibration file).

---

## Calibration artefacts

- **Nested gating:** `calibration/nested_edge_delta_thresholds.json`
  - Metadata keys: `run_name`, `timestamp_utc`, `git_sha`, `config_hash`, `trials_per_scenario`, `target_fpr`, `achieved_fpr`, `operating_points`.
  - Thresholds mirrored under both `thresholds` and `design_thresholds.nested`.
  - `lookup_calibrated_delta` is design-strict; missing design returns `None` (handled as `calibration_missing_p_T` skip).

- **Defaults:** `calibration_defaults.json` (energy-floor ROC for overlay; updated 2025-11-21).

Use `--gating-calibration calibration/nested_edge_delta_thresholds.json --gating-mode calibrated --design nested` in weekly runs to consume the nested grid.

---

## Logging & reproducibility

Follow `docs/DOCS_AND_LOGGING_SYSTEM.md`:
- Every run keeps `docs/agent_runs/<RUN_NAME>/{PROMPT,COMMANDS,RESULTS,TESTS,META}.md`.
- Update `PROGRESS.md` with branch, SHAs, commands, key metrics, and artefact paths.
- Required bundle contents enforced by `make gpt-bundle`.

---

## Stop-the-line rules (summary)

From `AGENTS.md`:
- No silent solver fallbacks; missing solvers must fail loud or be marked `skipped=true` with reason.
- Diagnostics cannot be opaque (`guard_other`, `diagnostic_failure` must carry context).
- Capped/truncated runs must be labeled and not used for headline claims.
- Never hand-edit `data/*.csv`; use ingest scripts.
- Merges require tests + logs at minimum (`make test-fast`).

---

## Troubleshooting tips

- **Missing calibration entry**: If you see `calibration_missing_p_T` in smoke outputs, regenerate calibration covering the observed `(p, T)` grid or lower `--assets-top` / adjust window to stay inside the grid.
- **cvxpy missing**: Set `FJS_FORCE_MISSING_CVXPY=1` to exercise skip paths with `mv_skip_on_missing_solver`; otherwise installs via dev extras.
- **Thread determinism**: Always set `EXEC_MODE=deterministic` for reproducible numbers; the runners will also cap OpenMP/BLAS threads.
- **Data digests mismatch**: Run `tools/verify_dataset.py <csv> --registry <registry.json>` to confirm hashes; update registry via `tools/update_registry.py` when refreshing datasets.

---

## Contributing

1. Work on a feature branch; keep commits small. Commit messages **must** include a “Tests run:” line with exact commands executed.
2. Run `make test-fast` (and any relevant smokes) before committing.
3. Capture commands and results in a new `docs/agent_runs/<RUN_NAME>/`.
4. If you add configs/knobs, document them in `project_state/CONFIG_REFERENCE.md` and ensure resolved configs write the new fields.
5. Build a bundle for review: `make gpt-bundle TICKET=<ticket> RUN_NAME=<run_name>`.
