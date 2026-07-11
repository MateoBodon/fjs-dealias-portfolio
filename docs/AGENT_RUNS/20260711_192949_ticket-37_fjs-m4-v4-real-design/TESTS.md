# Tests

- Focused v4 contract suite — PASS (`5 passed`).
- Ruff on v4 source, freezer, and tests — PASS.
- Python compile on v4 source, freezer, and tests — PASS.
- Bounded complete January 2013 source proof — PASS.
- Exact source receipt/size/content-hash readback — PASS.
- Exact-only analytical duplicate collapse — PASS (`2` collapsed).
- Conflicting duplicate rejection — PASS in unit coverage.
- Past-only factor-fit boundary — PASS (`fit_end < window_start`).
- Cell and manifest independent digest/readback validation — PASS.
- Independent real-source regeneration reproduced cell and manifest bytes.
- V2 full/smoke SHA preservation — PASS.
- V3 full/smoke SHA preservation — PASS.
- 2025 holdout refusal — PASS.
- Legacy ticker CSV exclusion — PASS.
- Combined v3/v4 contract suite — PASS (`75 passed`).
- `make detector-reference-gate` — PASS (`issue_count=0`).
- `make test-fast` — PASS (`191 passed, 188 deselected`).
- Native `pytest -m 'unit or integration' -q` — PASS.
- `make validate-runlogs` — PASS, including this run log.
- Canonical-root `make check-data-policy` — PASS.
- `git diff --check`, touched-path compile, and sensitive-path scan — PASS.
- Project OS strict verification — PASS
  (`val_20260711T193751470372Z_d7eee6bb`, three commands, containment safe,
  adequate for the active goal).

No restricted raw input or proof-cell payload was added to Git.
