# Tests

- [x] `. .venv/bin/activate && make validate-runlogs`
  - First run (expected FAIL before PROGRESS errata): detected ticket-34 final-stamp mismatch (`final=20260216_233000`).
- [x] `. .venv/bin/activate && pytest -q tests/test_validate_runlog.py tests/test_gpt_bundle.py tests/test_gpt_bundle_diff.py`
  - PASS (`10 passed`).
- [x] `. .venv/bin/activate && make validate-runlogs`
  - PASS after PROGRESS errata append.
- [x] `. .venv/bin/activate && make test-fast`
  - PASS (`87 passed, 171 deselected in 22.55s`).
- [x] `. .venv/bin/activate && BUNDLE_STAMP=20260217_011000 make gpt-bundle TICKET=35 RUN_NAME=20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance`
  - PASS (bundle created under `artifacts/_local/gpt_bundles/`).
- [x] `unzip -l artifacts/_local/gpt_bundles/20260217_011000_35_20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance.zip`
  - PASS (bundle contains run log, DIFF.patch, BUNDLE_META.md, and changed markdown snapshots).
