# Results

## Result

- Added a production-independent balanced-design reference oracle for FJS
  equation (5.5), stationary edges, admissible roots, component mapping, and
  spectral reconstruction.
- Froze exact scalar and two-stratum one-way values, invariances, and a
  deterministic magnitude-matched orthogonal sham paired with the planted
  oracle.
- Reclassified the historical Ticket 24 flat-zero curve from target-power
  evidence to an exact-config negative control: the hash-bound historical
  source used iid observation-level injection and recorded no component mode.
- Target-component power reduction now fails loudly unless every curve row
  carries the expected `inject_mode`; acceptance above detection is also a
  stop-line failure.
- Added `make detector-reference-gate`. It intentionally blocks with five
  issues: one-way bulk dimension, inclusion order, explicit-`C_s` MP mapping,
  spectral reconstruction, and missing target-power treatment provenance.
- Three strict expected failures keep known production mismatches visible. They
  are not waivers and do not permit calibration or broad empirical execution.
- The native fast suite passed with 106 tests and three intentional strict
  expected failures. Lint, compile, run-log, and registered data-policy checks
  passed.

## Claim boundary

This is a deterministic mechanism-correctness and provenance milestone, not an
empirical performance result. It used no restricted-data copy and launched no
synthetic headline, real-data grid, or memory-heavy job.

## Continuation

Repair production in the frozen order: one-way `N`/inclusion/mean-square
semantics; one consistent explicit-`C_s` equation (5.5) edge/root/t path;
reachable root solving; and defined spectral replacement semantics. Then make
the deterministic gate pass unchanged before target-matched between-component
null/power/invariance calibration.

Substantive implementation checkpoint:

- commit: `ce147d91305155e5d3d7c178465d8e63713ce343`
- tree: `755ed46510939929924ec7fc11871236ddc96082`
- parent: `e5399c4e94148abdad2a6585df9e548db9100025`

The final evidence commit and remote commit/tree readback are reported directly
to the Portfolio Administrator after the scoped push. No routine handoff/review
bundle is produced.
