# FJS Between-Component Mechanism Fixture Predeclaration

status: frozen before generator implementation or execution  
frozen_at: 2026-07-10T23:20:00Z  
project_os_goal: `goal_66cfb5598280`  
purpose: deterministic mechanism calibration only

## Claim boundary

This fixture may establish only that the repaired production detector can pass
the already-frozen deterministic reference/provenance gate on a small, explicit
balanced one-way mechanism. It is synthetic calibration, not real-data evidence,
not a null-size study, not a production power estimate, and never a headline
empirical result.

The historical Ticket 24 files remain immutable exact-configuration evidence.
This fixture is a new generation and may not be used to fill their missing
injection provenance.

## Frozen generation

The machine-readable authority is
`docs/artifacts/detector-contract-reference/between_mechanism_v1/input_spec.json`.
Its fixed design is:

- NumPy `PCG64` master seed `20260710`, with 12 child `SeedSequence` streams.
- Twelve paired trials at `mu` in `{0.0, 6.0}`.
- Balanced one-way design: 60 groups, 3 replicates, 10 features.
- Per trial, draw one standard-Normal direction and normalize it, one vector of
  60 standard-Normal group scores, and one `60 x 3 x 10` standard-Normal
  residual tensor.
- Reuse those exact draws across the two `mu` cells:
  `Y(mu) = sqrt(mu) * score * direction + 0.3 * residual`.
- The treatment label is exactly `inject_mode=between`.

The production detector configuration is fixed at `q_max=1`, `delta=0.3`,
`eps=0.03`, `stability_eta_deg=0.4`, `a_grid=60`, strict gating, isolated FJS
candidates only, off-component cap `0.3`, and `edge_mode=scm`. No coarse,
oracle, or sham candidate may enter this curve.

For each trial/cell, detection means at least one production FJS candidate
before overlay gating; acceptance means at least one retained candidate after
strict gating. Acceptance must remain a subset of detection. `curve.csv`
contains their means and exact integer counts.

## Frozen decision

The existing `assess_power_curve` thresholds and reference-value checks may not
change. The unchanged gate passes only if the repaired production reference
checks are exact, null detection is at most 7.5%, strong detection and
acceptance are each at least 80%, the detection gain is at least 50 percentage
points, the two-cell curve is nondecreasing, and treatment provenance is
`between`.

If the frozen fixture fails, this milestone reports the failed cell and stops.
It does not change the seed, trial count, dimensions, spike strength, detector
settings, or reducer after observing outputs.

## Required outputs

The deterministic generator must create:

- `curve.csv` — two aggregate rows;
- `trials.csv` — 24 paired trial/cell rows with child-stream identifiers;
- `manifest.json` — exact input-spec, generator, curve, and trial hashes plus
  the executing commit and environment versions.

A reproduction check must regenerate into a temporary directory and prove
byte-identical curve/trial outputs plus manifest-bound hashes.
