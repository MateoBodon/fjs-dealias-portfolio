# Tests

- `make test-fast` (failed: pytest not found)
- `make setup` (failed: externally managed environment)
- `. .venv/bin/activate && make test-fast` (pass)
- `. .venv/bin/activate && pytest -q tests/experiments/test_inject_spike.py` (pass)
