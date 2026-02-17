# Prompt

Ticket: **35**
Run: **20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance**
Summary: Fix ticket-34 canonical bundle provenance and enforce final BUNDLE_STAMP consistency

## Goal
- Make ticket-34 bundle provenance unambiguous in `PROGRESS.md` and enforce validator checks so future multi-stamp runs must reference the final bundle stamp.

## Constraints
- No new top-level directories.
- Keep bulky outputs under `artifacts/_local/` only.
- Keep `PROGRESS.md` append-only.
- Ensure `make validate-runlogs` and `make test-fast` pass.

## Plan
1. Add append-only PROGRESS errata naming ticket-34 canonical 233000 bundle and superseding 231243.
2. Harden `tools/agentic/validate_runlog.py` to enforce final stamp provenance for multi-stamp runs (with historical cutoff guard).
3. Document the rule, run tests/validation, and generate a ticket-35 review bundle.

## Files touched
- `PROGRESS.md`
- `tools/agentic/validate_runlog.py`
- `tests/test_validate_runlog.py`
- `Makefile`
- `tests/test_gpt_bundle.py`
- `docs/DOCS_AND_LOGGING_SYSTEM.md`
