# Claims And Evidence

last_updated: 2026-07-03
updated_by: Codex T-000
source_event: T-000 install AI Project OS v2

This repo makes research, benchmark, and reproducibility claims. Treat every claim below as bounded by the listed evidence and caveats.

| Claim | Status | Evidence | Validation / Caveat | Last Verified |
|---|---|---|---|---|
| The repo has strong auditability and reproducibility infrastructure. | Supported for engineering process | `AGENTS.md`, `PROGRESS.md`, `docs/agent_runs/`, `tools/agentic/`, `tests/test_gpt_bundle.py`, `tests/test_validate_runlog.py`. | Process strength does not imply research effect validity. | 2026-07-03 T-000 inspection |
| T-012 daily DoW four-leg matrix was recovered and is scientifically useful. | Partially supported | `project_state/CURRENT_RESULTS.md`; `docs/artifacts/rc-t-012/`; recovered local tree referenced at `/Volumes/Storage/Projects/fjs/_recovery/recovered_artifacts/rc-t-012`. | Review failed on monitoring/audit preservation; do not call it cleanly approved before ratification. | 2026-07-03 T-000 inspection |
| Daily DoW evidence is empirical-only and not detector validation. | Supported as current claim boundary | `project_state/CURRENT_RESULTS.md`; `project_state/KNOWN_ISSUES.md`; `docs/strategy/CONTEXT_CARRYOVER.md`. | Pro should decide future framing after state audit. | 2026-07-03 T-000 inspection |
| Weekly/oneway detector path is blocked by flat-zero injection sensitivity. | Supported as current blocker | `project_state/CURRENT_RESULTS.md`; `project_state/KNOWN_ISSUES.md`; cited artifact `reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv`. | T-000 did not rerun the experiment; evidence is from existing docs/artifacts. | 2026-07-03 T-000 inspection |
| Nested calibration coverage improved for p=188 and p=200 at T in {60,70,80}. | Supported for cited calibration artifact | `project_state/CURRENT_RESULTS.md`; `calibration/nested_edge_delta_thresholds.json`; `reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/run.json`. | Nested real-data smoke still has zero detections and is not a headline path. | Existing artifact snapshot |
| Capped/truncated runs are not headline evidence. | Supported policy | `AGENTS.md`; `docs/PLAN_OF_RECORD.md`; `project_state/VALIDATION_MATRIX.md`. | Requires continued enforcement in summary tooling and reviews. | 2026-07-03 T-000 inspection |
| T-000 installed AI Project OS v2 without product behavior changes. | Review-pending | T-000 diff, run log, archive manifest, and review bundle. | Heavy should inspect changed files and confirm only docs/tooling changed. | 2026-07-03 T-000 |

## Claims That Require Pro/Heavy Review Before Reuse

- Any advisor-facing statement that daily DoW is a durable performance improvement.
- Any statement that T-012 is approved rather than recovered/pending ratification.
- Any claim that the FJS/MANOVA detector is validated on realistic weekly/oneway financial windows.
- Any claim based on capped, truncated, or comparison-invalid runs.

## Evidence Hygiene Rules

- Cite artifact paths and validation commands with numerical claims.
- State whether a run is capped, uncapped, comparison-valid, and acceptance/detection-active.
- Keep raw data out of review/state bundles unless a ticket explicitly requires it.
- Preserve recovered/local-only paths as references, not as bundled data.
