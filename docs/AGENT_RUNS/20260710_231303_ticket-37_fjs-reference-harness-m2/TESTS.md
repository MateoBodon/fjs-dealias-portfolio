# Tests

- `python3 -m py_compile src/fjs/reference_oracle.py
  tools/check_fjs_reference.py` — PASS.
- `ruff check` over all changed Python paths — PASS.
- Focused reference and detector contract — PASS (`12 passed, 3 xfailed`).
- Expanded MP/edge/root/theta/balanced/de-alias/overlay surface — PASS with the
  same three intentional strict expected failures.
- `make test-fast` — PASS (`106 passed, 178 deselected, 3 xfailed`).
- `git diff --check` — PASS.
- `make detector-reference-gate` — EXPECTED BLOCK, five stable issue codes:
  `oneway_bulk_dimension_mismatch`, `oneway_inclusion_order_mismatch`,
  `explicit_cs_mp_map_mismatch`, `spectral_reconstruction_mismatch`, and
  `target_power_provenance_invalid`.
- `make check-data-policy` from the canonical SSD repo — PASS.
- `make validate-runlogs` — PASS after adding and validating this run's complete
  metadata and evidence files; historical legacy-META warnings remain
  non-fatal.

The three XFAILs are strict and expected to turn XPASS/fail if production
behavior changes without updating the frozen reference review. Broad empirical
execution remains prohibited while the deterministic gate is blocked.
