# Tests

## 1) Run-log validation
Command:
```bash
. .venv/bin/activate && make validate-runlogs
```
Result: **PASS** (exit 0)
Notes:
- `python3 tools/agentic/validate_runlog.py --all --repo .`
- New run log validated: `OK: .../docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh`
- Existing legacy run logs emit non-failing warnings (`uses legacy META.md; prefer META.json`).

## 2) Fast test suite
Command:
```bash
. .venv/bin/activate && make test-fast
```
Result: **PASS** (exit 0)
Output summary:
```text
pytest -m "unit"
83 passed, 171 deselected in 22.46s
```
