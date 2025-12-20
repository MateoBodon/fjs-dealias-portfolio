# fjs-dealias-portfolio

Robust covariance forecasting and portfolio gating over balanced equity panels, with synthetic calibration harnesses, weekly/daily evaluation runners, and reproducible reporting.

---

## Current Status (2025-12-20)

- **Tests:** `make test-fast` (Python 3.12, .venv) is green; keep this as the minimum gate for any commit.
- **Latest RC-lite drop:** `reports/rc-20251121/` (DoW/vol, Tyler edge, FF5+MOM, 126×21 windows capped at first 200). Detection ≈4.3%, ΔMSE ~1e-13 scale; manifests/regime CSV included.
- **Nested calibration (ticket-10):** `calibration/nested_edge_delta_thresholds.json` (run `20251220_011519_ticket-10_nested-null-fpr`, git `e6e7982`). Null detections 0/220 (Wilson hi 0.017), power=1.0 at delta_frac=0.05; metadata embeds run_name/timestamp/git_sha/config_hash/trials/operating_points and is mirrored under `design_thresholds.nested`.
- **Nested real-data smoke (ticket-14 fixup):** `make run:equity_nested_smoke_tiny` executes 3 capped windows (p≈188, T=70/80). All skipped with explicit `calibration_missing_p_T`; guard tallies stability_fail=3, others=0. delta_frac falls back to config (0.008). Outputs: `experiments/equity_panel/outputs_nested_smoke_tiny/`.
- **Known gaps:** Nested calibration grid lacks p≈188 coverage; guard attribution cleanup (ticket-07) still open; see `project_state/KNOWN_ISSUES.md`.

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

