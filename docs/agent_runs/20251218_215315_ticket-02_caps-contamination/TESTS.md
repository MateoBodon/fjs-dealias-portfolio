# Tests
- 2025-12-18: PATH=.venv/bin:$PATH make test-fast (pass; 67 passed, 147 deselected; pytest DeprecationWarning about datetime.utcnow).
- 2025-12-18: PATH=.venv/bin:$PATH make test-fast (rerun after summary/manifest tweaks; pass with same counts).
- Earlier attempt `make test-fast` failed because pytest was not installed on PATH; resolved by creating .venv and installing dev extras.
