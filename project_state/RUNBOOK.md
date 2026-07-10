# Runbook

last_updated: 2026-07-03
updated_by: Codex T-000
source_event: T-000 install AI Project OS v2

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
make setup
```

`make setup` upgrades pip and installs the project with dev extras from `pyproject.toml`.

## Minimum Local Validation

```bash
. .venv/bin/activate
make test-fast
```

This is the repo's minimum commit gate. It runs `pytest -m "unit"`.

## Targeted Validation

```bash
. .venv/bin/activate
pytest -q tests/test_ai_os_bundle.py tests/test_gpt_bundle.py
python3 -m json.tool docs/_archive/pre_ai_os_v2/20260703/ARCHIVE_MANIFEST.json >/dev/null
```

Use targeted tests for doc/tooling tickets when the changed surface is narrow.

## Run-Log and Data-Policy Checks

```bash
. .venv/bin/activate
make validate-runlogs
make check-data-policy
```

Run these when changing agent run logs, bundle policy, data handling, or repo hygiene rules.

## AI OS v2 Bundles

Project State Audit Bundle:

```bash
python3 tools/agentic/ai_os_bundle.py --profile project_state_audit
# or
make project-state-audit-bundle
```

T-000/Heavy review bundle:

```bash
python3 tools/agentic/ai_os_bundle.py \
  --profile review \
  --ticket T-000 \
  --run-log reports/_runs/20260703_132437_T-000_install_ai_project_os_v2 \
  --state-bundle reports/_bundles/<stamp>_repo_project-state_initial.zip
```

Existing repo-specific GPT review bundle:

```bash
make gpt-bundle TICKET=<ticket> RUN_NAME=<run_name>
```

Keep this target for normal post-ticket review bundles; it preserves the existing merge-base diff and run-log validation behavior.

## Research Smokes

```bash
EXEC_MODE=deterministic make run:equity_nested_smoke_tiny
EXEC_MODE=deterministic make rc-lite-sanity
```

These can be more expensive than docs/tooling checks. Run them only when relevant to changed experiment code or research claims.

## Reporting / Summary

```bash
PYTHONPATH=src:. python tools/make_summary.py --rc-dir <reports/rc-dir>
```

## Safety Rules

- Do not hand-edit raw data CSVs.
- Do not include raw data or bulky generated report trees in AI bundles by default.
- Do not present capped/truncated runs as headline evidence.
- Record exact commands and outcomes in the ticket run log.
