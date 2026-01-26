# Tests

- `make test-fast && python3 tools/agentic/project_state_refresh.py --zip && python3 tools/agentic/repo_snapshot.py && git status --porcelain` (failed: pytest not found)
- `. .venv/bin/activate && make test-fast && python3 tools/agentic/project_state_refresh.py --zip && python3 tools/agentic/repo_snapshot.py && git status --porcelain` (pass; 78 passed, 170 deselected)
