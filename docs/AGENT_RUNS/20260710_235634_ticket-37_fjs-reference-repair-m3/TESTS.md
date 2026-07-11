# Tests

- Frozen independent reference oracle versus production MP/edge/root/component
  and reconstruction surface — PASS; no XFAIL remains.
- Focused MP/de-alias/theta/reconstruction/overlay suite — PASS.
- `make detector-reference-gate` — PASS (`issue_count=0`).
- Frozen fixture byte reproduction via `--check` — PASS.
- `make test-fast` — PASS (`117 passed, 178 deselected`).
- Combined native `unit or integration` suite — PASS.
- Equity-panel ablation integration path — PASS after bounded strongest-root
  theta seeding.
- `ruff check` over changed Python paths — PASS.
- `git diff --check` — PASS.
- `make check-data-policy` from the canonical SSD repo — PASS.
- `make validate-runlogs` — PASS after adding this run's complete metadata.

Five-process fixed-panel timing medians:

- repaired unbounded checkpoint `4437571`: `5.318868 s`;
- bounded checkpoint `9445a37`: `0.380708 s` (`13.971x`, `92.842%` reduction);
- strongest-first checkpoint `9afbb72`: `0.395707 s` (`13.441x`, `92.560%`
  reduction).

The 9af result is `3.940%` slower than 944 on this small fixed panel, so no
incremental speedup claim is made between those two commits. Its purpose is to
choose the strongest admissible root seeds before the exact solve, eliminating
fixture-order sensitivity while retaining the large bounded-search gain.

No broad grid, cloud execution, or real-data result was run.
