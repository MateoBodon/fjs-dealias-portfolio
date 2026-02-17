# Results

## Summary
- Corrected ticket-35 run metadata drift by updating `git_sha_after` to `71a700bb15a7f39b70a705215d5258e2d24549f3` in both canonical (`META.json`) and legacy (`META.md`) files.
- Added a new SHA guardrail path in `tools/agentic/validate_runlog.py` and wired `make gpt-bundle` to enforce it against bundled `head_sha` for timestamped runs (`>= 20260216_000000`).
- Added regression coverage for placeholder/mismatch cutoff behavior in `tests/test_validate_runlog.py` and made bundle-target assertions cover the new `--expected-head-sha` enforcement snippet in `tests/test_gpt_bundle.py`.
- Updated `docs/DOCS_AND_LOGGING_SYSTEM.md` fail-loud contract to document the new `META.json.git_sha_after` enforcement rule.

## Key outputs
- Code/doc changes:
  - `tools/agentic/validate_runlog.py`
  - `Makefile`
  - `tests/test_validate_runlog.py`
  - `tests/test_gpt_bundle.py`
  - `docs/DOCS_AND_LOGGING_SYSTEM.md`
  - `docs/agent_runs/20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance/META.json`
  - `docs/agent_runs/20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance/META.md`
- Bundle artifact:
  - `artifacts/_local/gpt_bundles/20260217_020630_36_20260217_020118_ticket-36_enforce-runlog-meta-sha-correctness-fix-ticket35-drift.zip`

## Notes
- Legacy run logs that only have `META.md` continue to emit warnings but remain non-fatal under `make validate-runlogs`.
- Superseded prior ticket-35 bundle path:
  - `artifacts/_local/gpt_bundles/20260217_011000_35_20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance.zip`
