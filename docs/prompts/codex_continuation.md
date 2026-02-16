# Codex Continuation Prompt (Canonical Start)

Use this prompt when resuming work in `fjs-dealias-portfolio`.

## Mission
Advance the project toward publishable, advisor-credible evidence for the FJS de-aliasing overlay.

## Required read order
1. `AGENTS.md`
2. `docs/PLAN_OF_RECORD.md`
3. `docs/CODEX_SPRINT_TICKETS.md`
4. `project_state/CURRENT_RESULTS.md`
5. `project_state/KNOWN_ISSUES.md`
6. Latest run logs under `docs/agent_runs/` (start with newest ticket run)

## Non-negotiables (hard gates)
- No silent fallbacks.
  - Missing config path: hard error.
  - Missing solver: fail-loud or explicit skip with reason.
- No headline claims from capped runs.
  - Any `cap_active=true` run is non-headline and must be labeled as such.
- Comparison validity is mandatory.
  - Report delta metrics and DM tests only on aligned window intersections.
  - If `comparison_valid_*` is false or `n_effective_*` is weak, call it out prominently.

## Current truth snapshot
- Pipeline quality is strong: tests, run logs, metadata, and bundle workflow are in place.
- Research is still blocked by evidence quality, not by missing engineering infrastructure.
- Two blockers dominate:
  1. Injection sensitivity on week-design real windows remains flat-zero.
  2. No advisor-ready uncapped week-design run has closed the headline gate.

## Immediate priorities (do these in order)
1. Ticket #18: injection flat-zero root cause + mini diagnostics.
   - Produce artifact-backed per-stage attribution for where detection dies.
   - Either show a non-flat `mu -> detection/acceptance` response or produce a hard, stage-specific explanation.
2. Ticket #20: one advisor-ready uncapped week run.
   - Require `cap_active=false`, comparison-valid metrics, meaningful `n_effective_*`, and complete detection/acceptance/skip reporting.
3. Only after Tickets #18 and #20 are closed, expand grid breadth.

## Execution protocol
- Start each run with `tools/agentic/runlog_init.py` and a valid `RUN_NAME`.
- Keep `PROGRESS.md` append-only with exact commands, test outcomes, and artifact paths.
- Minimum validation before claiming done:
  - `. .venv/bin/activate && make validate-runlogs`
  - `. .venv/bin/activate && make test-fast`
- For review handoff, generate:
  - `. .venv/bin/activate && make gpt-bundle TICKET=<NN> RUN_NAME=<RUN_NAME>`

## Fail-closed behavior
If required evidence cannot be produced, do not polish narratives. Log the blocker, attach artifact paths, and stop at the failed gate.
