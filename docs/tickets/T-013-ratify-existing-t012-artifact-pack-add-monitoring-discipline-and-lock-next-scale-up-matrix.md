# T-013: ratify existing T-012 artifact pack, add monitoring discipline, and lock next scale-up matrix

## Goal

Convert the recovered T-012 campaign from a scientifically useful but review-failed artifact pack into a clean base for the next larger cycle. Do not rerun completed T-012 eval legs unless an existing output is first shown corrupt.

## Why now

T-012 spent the heavy compute and produced a useful four-leg daily DoW empirical result. The review failure was operational: late monitoring checkpoints were not fully audit-backed in the preserved command log. The next work should preserve the science, repair the operating discipline, and lock the next bounded scale-up campaign.

## Scope

Allowed:

- ratify the recovered T-012 summary pack from existing artifacts
- write a tracked ratification memo that separates scientific output quality from the monitoring/audit review failure
- add reusable long-run monitoring discipline for future heavy evals
- update current state docs to the post-ratification truth
- lock the next larger daily DoW scale-up ticket

Excluded:

- no rerun of any completed T-012 eval leg unless corruption is proven first
- no detector, gating, estimator, portfolio, calibration, or data-source changes
- no claim that daily DoW is detector-validated
- no rewrite of T-012 history to pretend the original monitoring contract passed review

## Acceptance Criteria

- `docs/artifacts/rc-t-012/summary/` remains the curated T-012 summary surface.
- A ratification memo exists and states:
  - T-012 scientific outputs were not shown corrupt
  - the original review failure was monitoring/audit preservation
  - daily DoW remains empirical-only and not detector validation
- A reusable monitoring protocol or helper exists for future long evals.
- The next scale-up ticket is tracked and requires the monitoring protocol.
- Run-log validation and `make test-fast` pass.

## Required Tests

- `make validate-runlogs`
- `make test-fast`

## Invariants

- Keep T-012 artifacts stable.
- Keep heavy recovered details local-only unless a small curated surface is intentionally promoted.
- Preserve the empirical-only claim boundary.
