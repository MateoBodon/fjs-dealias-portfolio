# Tests

- [x] git status -sb
  - Result: output not captured in run log; bundle metadata recorded `git_dirty=false`.
- [x] git clean -ndx
  - Result: output not captured in run log; bundle metadata recorded `git_dirty=false`.
- [x] git status --porcelain
  - Result: output not captured in run log; bundle metadata recorded `git_dirty=false`.

- [x] pytest -q tests/test_repo_hygiene.py
  - Result: pass
- [x] make test-fast
  - Result: pass (83 passed, 171 deselected)
- [x] make validate-runlogs
  - Result: pass
- [x] make check-data-policy
  - Result: pass
