# Tests

- Focused M4 manifest/runner plus synthetic harness/calibration tests — PASS.
- Exact-binomial known-value tests — PASS.
- Simulator planted/nuisance direction replay — PASS.
- Full-manifest fail-closed and smoke-output isolation tests — PASS.
- Real two-process intentional interruption, resume, fresh replay — PASS;
  stable reducer SHA-256 matched and surviving worker count was zero.
- Manifest byte regeneration via `cmp` — PASS.
- `make detector-reference-gate` — PASS (`issue_count=0`).
- `make test-fast` — PASS (`117 passed, 187 deselected`).
- Native `pytest -m 'unit or integration' -q` — PASS (`rc=0`).
- Changed-path Ruff and Python compile — PASS.
- `git diff --check` — PASS.
- Canonical `make check-data-policy` — PASS.

The isolated worktree's data-policy command correctly reported its absent local
raw-data links; the canonical SSD repository passed. No raw or restricted data
was added to Git.
