# Ticket 37 — FJS Scientific Recenter Milestone 1

## Objective

Freeze the ambitious real-data decision contract and remove silent provenance
and selection fallbacks before any detector repair or broad empirical run.

## Acceptance criteria

- The predeclaration fixes the detector stop-line, exact real-data authorities,
  point-in-time universe, time splits, modern baseline ladder, holdout policy,
  endpoints, and claim reducer.
- `assets_top` fails unless an explicit dated ranking source is supplied and
  hashed into run metadata.
- Every candidate is labeled `fjs`, `coarse`, `oracle`, or `sham`; missing,
  unknown, or mixed provenance fails loudly.
- Unknown estimators and unrecoverable eigendecomposition failures do not
  silently become a different treatment.
- The historical flat-zero curve is restored only as a small hash-bound
  reference and fails the frozen detector reducer in a deterministic unit test.
- Targeted tests, `make test-fast`, run-log validation, and data-policy checks
  pass before commit/push.
- No raw restricted data, broad CRSP run, main merge, or public release occurs.

## Stop condition and continuation

This milestone is complete when the scoped SSD branch is pushed and its remote
commit/tree are read back. The broader scientific goal remains blocked at the
detector stop-line. The next bounded action is an independent deterministic
reference harness for the MP edge, roots, mapped component, and reconstructed
covariance.

