# Prompt

Ticket: **27**
Run: **20260212_230317_ticket-27_runlog-meta-json-alignment**
Summary: Align run-log schema and tooling to canonical `META.json`, and close ticket-27 audit evidence gaps.

## Goal
- [x] Resolve the audit FAIL by making run logs, validator/tooling, and bundle contract consistent with `META.json` requirements.

## Constraints
- [x] Tracking policy followed (no new top-level dirs; outputs in canonical zones)
- [x] No secrets in repo or logs
- [x] Tests run (or explicitly marked N/A)

## Plan
1. Verify cited gaps (`META.json` missing, bundle/diff mismatch, follow-up run-log presence).
2. Patch tooling/docs to remove `META.md`-only requirement ambiguity.
3. Add missing metadata files, run validations/tests, and regenerate bundle evidence.

## Files to touch (expected)
- AGENTS.md
- docs/DOCS_AND_LOGGING_SYSTEM.md
- Makefile
- tools/agentic/runlog_init.py
- tools/agentic/validate_runlog.py
- tools/agentic/README.md
- tests/test_gpt_bundle.py
- docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/*
- docs/agent_runs/20260127_222035_ticket-27_repo-hygiene-cleanup-followup/*
- docs/agent_runs/20260128_014550_ticket-27_runlog-audit-fix/*
- PROGRESS.md

## Definition of Done
- [x] `META.json` exists where required by the rubric
- [x] Tooling/docs clearly define canonical metadata file
- [x] `make validate-runlogs` and `make test-fast` pass
- [ ] Updated bundle generated from committed state
