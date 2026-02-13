# Tests

- [x] . .venv/bin/activate && make validate-runlogs
  - Result: pass. Warning: DeprecationWarning for datetime.utcnow().
- [x] . .venv/bin/activate && make test-fast
  - Result: pass (83 passed, 171 deselected). Warning: DeprecationWarning for datetime.utcnow().
