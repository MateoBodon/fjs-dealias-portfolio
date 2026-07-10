# Tests

- `python3 -m py_compile ...` — PASS.
- `pytest -q tests/fjs/test_detector_contract.py tests/fjs/test_overlay.py` —
  PASS with one intentional strict XFAIL for the historical flat-zero curve.
- Targeted ranked-universe subset — PASS (`4 passed`) after adding panel/input
  provenance to `run.json`.
- Final focused surface excluding one unchanged base failure — PASS (78
  collected; one intentional strict XFAIL).
- Full focused surface without exclusion — FAIL only at
  `test_holdout_empty_windows_do_not_trigger_window_coverage_cap`; the exact
  failure reproduces at pinned base `193a325d` because the fallback loader's
  unchanged 22-row minimum exceeds the fixture's 14 rows.
- `make test-fast` — PASS (`97 passed, 178 deselected, 1 xfailed`).
- `pytest -m 'unit or integration' -q` — PASS (112 collected, including one
  intentional strict XFAIL).
- `ruff check src/fjs/detector_contract.py tests/fjs/test_detector_contract.py`
  — PASS.
- `ruff check --select F,E9` over all changed Python paths — PASS.
- `git diff --check` — PASS.
- `make validate-runlogs` — PASS (legacy warnings only).
- `make check-data-policy` from the isolated worktree — expected FAIL because
  ignored local return files are not copied into a new worktree; the exact
  failure reproduces at the pinned base worktree.
- `make check-data-policy` from `/Volumes/Storage/Projects/fjs/repo` — PASS
  against registered local data, including legacy returns SHA-256
  `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`.

No broad empirical test was run by design.
