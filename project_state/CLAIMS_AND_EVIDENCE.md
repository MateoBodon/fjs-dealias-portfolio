# Claims And Evidence

last_updated: 2026-07-10
updated_by: Portfolio OS FJS worker
source_event: Ticket 37 scientific recenter milestone 1

This repo makes research, benchmark, and reproducibility claims. Treat every claim below as bounded by the listed evidence and caveats.

| Claim | Status | Evidence | Validation / Caveat | Last Verified |
|---|---|---|---|---|
| The repo has strong auditability and reproducibility infrastructure. | Supported for engineering process | `AGENTS.md`, `PROGRESS.md`, `docs/agent_runs/`, `tools/agentic/`, `tests/test_gpt_bundle.py`, `tests/test_validate_runlog.py`. | Process strength does not imply research effect validity. | 2026-07-03 T-000 inspection |
| T-012 daily DoW four-leg matrix was recovered and is scientifically useful only as historical coarse-overlay evidence. | Partially supported | `project_state/CURRENT_RESULTS.md`; `docs/artifacts/rc-t-012/`; recovered local tree referenced at `/Volumes/Storage/Projects/fjs/_recovery/recovered_artifacts/rc-t-012`. | All 6,917 changed full-regime windows are `coarse`; review also failed on monitoring/audit preservation. It is not an FJS effect. | 2026-07-10 Ticket 37 audit |
| T-012 does not beat the best implemented baseline and is not promotable. | Supported for the recovered four-leg matrix | `docs/artifacts/rc-t-012/summary/t012_full_regime_comparison.csv`; leg diagnostics in the recovery tree. | Overlay loses to the best CC/EWMA comparator in all eight leg-by-portfolio QLIKE comparisons; EW MSE worsens in all four legs. | 2026-07-10 Ticket 37 audit |
| Weekly/one-way detector path is blocked by flat-zero injection sensitivity. | Supported as current blocker | `docs/artifacts/detector-contract-reference/ticket24_week_full_fix/{manifest.json,curve.csv,gating_reasons.csv}`; `tests/fjs/test_detector_contract.py`. | The hash-bound historical curve has zero detection and acceptance at every tested spike. It falsifies only that exact configuration, not every possible repair. | 2026-07-10 hash/readback and deterministic unit test |
| Candidate provenance is now fail-loud. | Supported for code contract | `src/fjs/detector_contract.py`; `src/fjs/dealias.py`; `src/fjs/overlay.py`; `tests/fjs/test_overlay.py`. | `fjs`, `coarse`, `oracle`, and `sham` must remain separate arms; unknown/missing/mixed sources raise. | 2026-07-10 Ticket 37 tests |
| Alphabetical `assets_top` selection is prohibited. | Supported for daily runner contract | `experiments/eval/run.py`; `experiments/eval/config.py`; `tests/experiments/test_eval_run.py`. | A dated ranked snapshot and hashes are now required, but the rolling PERMNO/lagged-cap CRSP adapter is still missing and required for the flagship. | 2026-07-10 Ticket 37 tests |
| The ambitious flagship design is frozen before broad execution. | Supported as predeclaration | `docs/strategy/FJS_SCIENTIFIC_RECENTER_PREDECLARATION.md`. | The detector gate currently fails; no real-data performance result has been produced by Ticket 37. | 2026-07-10 |
| Nested calibration coverage improved for p=188 and p=200 at T in {60,70,80}. | Supported for cited calibration artifact | `project_state/CURRENT_RESULTS.md`; `calibration/nested_edge_delta_thresholds.json`; `reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/run.json`. | Nested real-data smoke still has zero detections and is not a headline path. | Existing artifact snapshot |
| Capped/truncated runs are not headline evidence. | Supported policy | `AGENTS.md`; `docs/PLAN_OF_RECORD.md`; `project_state/VALIDATION_MATRIX.md`. | Requires continued enforcement in summary tooling and reviews. | 2026-07-03 T-000 inspection |
| T-000 installed AI Project OS v2 without product behavior changes. | Review-pending | T-000 diff, run log, archive manifest, and review bundle. | Heavy should inspect changed files and confirm only docs/tooling changed. | 2026-07-03 T-000 |

## Claims That Require Pro/Heavy Review Before Reuse

- Any advisor-facing statement that daily DoW is a durable performance improvement.
- Any statement that T-012 is an FJS result or approved rather than
  coarse-attributed/recovered/pending ratification.
- Any claim that the FJS/MANOVA detector is validated on realistic weekly/oneway financial windows.
- Any claim based on capped, truncated, or comparison-invalid runs.
- Any claim using the static snapshot universe, exposed 2024 bridge, or unopened
  2025 holdout as if it were the frozen point-in-time flagship result.

## Evidence Hygiene Rules

- Cite artifact paths and validation commands with numerical claims.
- State whether a run is capped, uncapped, comparison-valid, and acceptance/detection-active.
- Keep raw data out of review/state bundles unless a ticket explicitly requires it.
- Preserve recovered/local-only paths as references, not as bundled data.
- Bind exact candidate source, point-in-time membership, dataset manifest, input
  hashes, code commit, baseline implementation, and paired window set.
