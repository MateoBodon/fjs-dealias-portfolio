# Tests

- [x] `pytest -q tests/test_validate_runlog.py tests/test_gpt_bundle.py tests/test_gpt_bundle_diff.py`
  - Result: PASS (`13 passed`).
- [x] `. .venv/bin/activate && make validate-runlogs`
  - Result: PASS.
- [x] `. .venv/bin/activate && make test-fast`
  - Result: PASS (`90 passed, 171 deselected`).
