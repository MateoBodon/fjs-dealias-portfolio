# Tests
- 2025-12-19: `source .venv/bin/activate && make test-fast` (pass; 67 passed, 144 deselected; initial attempt timed out at 10s, reran with extended timeout).
- 2025-12-19: `source .venv/bin/activate && pytest -m unit -k "summary or summarize_rc_sanity or run_meta"` (pass; 4 passed, 207 deselected).
