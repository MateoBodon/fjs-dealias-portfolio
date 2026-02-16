# Tests

## 1) Run-log validation
- Command: `. .venv/bin/activate && make validate-runlogs`
- Result: PASS
- Notes: legacy runs emit non-fatal `META.md` warnings as expected.

## 2) Fast test suite
- Command: `. .venv/bin/activate && make test-fast`
- Result: PASS
- Output summary: `83 passed, 171 deselected in 22.53s`.
