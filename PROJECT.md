# PROJECT.md

## Purpose
`fjs-dealias-portfolio` is a research codebase for testing whether an FJS/MANOVA-style de-aliasing overlay can improve covariance forecasts and downstream portfolio risk decisions on balanced equity panels.

## What This Project Is
- Type: research + evaluation infrastructure
- Primary language: Python
- Core outputs: run artifacts under `reports/`, research summaries, and auditable run logs under `docs/agent_runs/`

## Current State (2026-07-03)
- AI Project OS v2 is installed for strategy, review, run-log, archive, and bundle workflows.
- Engineering pipeline is strong: run logging, validation checks, and test gates are in place and routinely used.
- Synthetic calibration and real-data evaluation runners are operational.
- Research evidence is not yet publishable without further claim review.
- Current recovered high-water mark: T-012 daily DoW four-leg empirical matrix, treated as scientifically useful but not cleanly ratified because the monitoring/audit trail failed.
- Weekly/oneway detector validity remains blocked by flat-zero injection-sensitivity evidence.

## Biggest Risks
- Detection/gating mathematics may not map cleanly onto financial residual structure in real windows.
- Grid expansion before fixing injection flat-zero would create more artifacts without resolving the core validity question.
- Headline claims can drift if capped or comparison-invalid runs are not strictly excluded.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make test-fast
```

## Where Outputs Go
- Run logs: `docs/agent_runs/<RUN_NAME>/`
- T-000 / AI OS v2 run logs: `reports/_runs/<RUN_NAME>/`
- Research outputs: `reports/<run_dir>/`
- Pro/Heavy bundles: `reports/_bundles/`
- Local bulky artifacts/scratch bundles: `artifacts/_local/`
- Pre-v2 documentation archive: `docs/_archive/pre_ai_os_v2/20260703/`

## Canonical AI OS v2 Docs
- Strategy/current context: `docs/strategy/`
- Ticket template and ledger inputs: `docs/tickets/`
- Factual state map: `project_state/STATE_INDEX.md`
- Validation map: `project_state/VALIDATION_MATRIX.md`
- Claim surface: `project_state/CLAIMS_AND_EVIDENCE.md`

## What Counts As Done
A research milestone is done only when all are true:
- `make test-fast` passes.
- Run log is complete (`PROMPT.md`, `COMMANDS.md`, `RESULTS.md`, `TESTS.md`, `META.json`).
- Results are uncapped for headline use (`cap_active=false`) and comparison-valid (`comparison_valid_*` true with meaningful `n_effective_*`).
- Injection sensitivity is non-flat for at least one design, or a defensible explanation for flat-zero behavior is documented with artifact evidence.
- `PROGRESS.md` and `project_state/*` are updated with artifact-backed claims only.
