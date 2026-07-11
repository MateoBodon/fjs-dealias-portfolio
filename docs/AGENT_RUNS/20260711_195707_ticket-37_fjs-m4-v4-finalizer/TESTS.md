# Tests

- Focused v4 finalizer suite — PASS (`4 passed`).
- Synthetic 72-source/72-cell complete checkpoint — PASS.
- Exact restart idempotence and atomic reload — PASS.
- Missing/duplicate/cross-generation/identity rejection — PASS.
- Stable aggregate manifest and independent readback — PASS.
- Source/cell aggregate digest checks — PASS.
- Cell artifact tamper rejection — PASS.
- CLI init/register/status and incomplete-finalize stop — PASS.
- Explicit full-execution/AWS/outcome/2025 false boundaries — PASS.
- Ruff and Python compile for new finalizer surfaces — PASS.
- Combined v3/v4/finalizer contract suite — PASS (`79 passed`).
- `make test-fast` — PASS (`195 passed, 188 deselected`).
- Native `pytest -m 'unit or integration' -q` — PASS.
- `make detector-reference-gate` — PASS (`issue_count=0`).
- `make validate-runlogs` — PASS, including this run log.
- Canonical-root `make check-data-policy` — PASS.
- Project OS strict verification — PASS
  (`val_20260711T200220822211Z_04730216`, containment safe and adequate for
  the active goal).

No raw source, generated cell, outcome, or full final manifest was added to
Git.
