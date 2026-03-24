# Tests

- [x] `pytest -q tests/test_validate_runlog.py tests/test_gpt_bundle.py tests/test_gpt_bundle_diff.py`
  - Result: PASS (`13 passed`).
- [x] `. .venv/bin/activate && make validate-runlogs`
  - Result: PASS.
- [x] `. .venv/bin/activate && make test-fast`
  - Result: PASS (`90 passed, 171 deselected`).
- [x] `. .venv/bin/activate && BUNDLE_STAMP=20260217_020630 make gpt-bundle TICKET=36 RUN_NAME=20260217_020118_ticket-36_enforce-runlog-meta-sha-correctness-fix-ticket35-drift`
  - Result: PASS (bundle created; runlog `git_sha_after` guardrail satisfied).
- [x] `. .venv/bin/activate && make validate-runlogs` (post-finalization rerun)
  - Result: PASS.
- [x] `. .venv/bin/activate && make test-fast` (post-finalization rerun)
  - Result: PASS (`90 passed, 171 deselected`).
