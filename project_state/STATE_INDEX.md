# State Index

last_updated: 2026-07-03
updated_by: Codex T-000
source_event: T-000 install AI Project OS v2

## Repository Identity

`fjs-dealias-portfolio` is a Python research codebase for evaluating whether an FJS/MANOVA-style de-aliasing overlay can improve covariance forecasts and downstream portfolio-risk decisions on balanced equity panels.

## Major Areas

| Path | Purpose | Current Status |
|---|---|---|
| `src/` | Core reusable package code: FJS detection/gating/overlay, baselines, finance utilities, IO, plotting, reporting. | Product/research code; not changed by T-000. |
| `experiments/` | Daily, weekly/equity-panel, synthetic, and evaluation runners/configs. | Operational; expensive runs should be strategy-gated. |
| `tests/` | Unit/integration/slow test surface for algorithms, runners, bundle tooling, and repo hygiene. | `make test-fast` is the minimum commit gate. |
| `tools/` | Reporting, bundle, state, monitoring, summary, and utility scripts. | T-000 adds `tools/agentic/ai_os_bundle.py`. |
| `docs/strategy/` | AI Project OS v2 canonical strategy/current context docs. | Current strategic surface after T-000; Pro should update after state audit. |
| `docs/tickets/` | Human/agent work orders. | T-000 and template added; older tickets preserved. |
| `docs/agent_runs/` | Historical Codex run logs. | Pre-v2 historical source; copied into archive snapshot. |
| `docs/_archive/pre_ai_os_v2/20260703/` | Pre-v2 archive index, manifest, and copied snapshots. | Current archive for T-000. |
| `project_state/` | High-signal factual memory: architecture, dataflow, results, issues, validation, claim/evidence docs. | Existing docs preserved; T-000 adds v2 state docs. |
| `reports/` | Research outputs, memos, summaries, generated artifacts, and new T-000 bundles. | Large/generated outputs are mostly indexed rather than bundled. |
| `data/` | Local/registered sample and research data. | Raw data excluded from AI bundles by default. |
| `calibration/` | Calibration thresholds/defaults and related plots/metadata. | Product input; do not move into archive. |

## Current Research State

- The daily DoW empirical lane has recovered T-012 evidence and curated summary surfaces under `docs/artifacts/rc-t-012/`.
- T-012 is not cleanly ratified because review failed on monitoring/audit preservation.
- Weekly/oneway detector validation remains blocked by flat-zero injection sensitivity in current week-design evidence.
- Existing docs emphasize no headline claims from capped, truncated, comparison-invalid, or acceptance-zero runs.

## Current Documentation State

- Current v2 strategy docs: `docs/strategy/`.
- Current factual state docs: `project_state/`.
- Current chronological log: `PROGRESS.md`.
- Pre-v2 historical docs: `docs/_archive/pre_ai_os_v2/20260703/` plus original paths.

## Bundle State

- Project State Audit Bundles: `reports/_bundles/*_project-state_initial.zip`.
- Review bundles: `reports/_bundles/*_review_T-000.zip`.
- T-000 run log: `reports/_runs/20260703_132437_T-000_install_ai_project_os_v2/`.

## Missing Or Awaiting Strategy

- Pro has not yet rewritten the goal contract or plan of record after reading the T-000 Project State Audit Bundle.
- Heavy has not yet reviewed the T-000 installation bundle.
- T-013 remains the likely next work item, but Pro/Heavy should confirm scope and priority.
