# FJS M5 Rolling 156-Week Geometry Contract

status: frozen before any real v5 geometry proof or detector outcome
as_of: 2026-07-11
active_goal: `goal_79bc96eb9b7f`

## Scientific purpose

M4 v4 completed exact monthly provenance, but aggregate QA rejected it as the
flagship realistic design: all 72 cells had zero missingness and only 9-13
residual dates. M5 is additive. It does not reinterpret or overwrite v2-v4.

M5 asks whether the FJS detector can satisfy the already frozen null, planted-
power, attribution, and invariance gates across actual rolling development
geometry. Geometry must be accepted before any detector outcomes are run.

## Frozen endpoint and window design

- Warm-up/input history begins `2010-01-01`.
- Development endpoints are the last registered FF6 calendar date of every
  month from `2013-01` through `2018-12` (72 endpoints).
- Each input window contains exactly 156 Monday-anchored calendar weeks ending
  at its formation date. Five weekday positions are the nominal replicates;
  holidays and security-specific absences remain observed geometry.
- Forward evaluation blocks are endpoint-to-next-endpoint, so the primary
  monthly blocks do not overlap.
- The only bounded real proof allowed in this lifecycle is `2013-01`, selected
  before inspecting v5 geometry. The full 72-endpoint corpus is forbidden
  until the separate portfolio memory-heavy lane is released.

## Point-in-time universe

Apply the frozen CIZ common-equity, exchange, active-trading, price, positive-
cap, and observed-return eligibility filters. Rank PERMNOs using the last
eligible `dlycap` no more than 10 calendar days before formation, descending,
with PERMNO ascending as the deterministic tie-break. A candidate needs at
least 390 past observations inside the rolling window. Select exactly 60.
There is no future backfill.

The source reader checks `dlyret` presence and numeric validity as part of the
frozen eligibility filter. It discards that column before geometry
construction; no return values, detector outputs, or performance summaries are
persisted in a v5 proof.

## Coverage and geometry gates

These thresholds are frozen before the `2013-01` proof and may not be tuned to
its result:

- exactly 156 week groups and at least 720 trading dates;
- exactly 60 selected assets;
- at least 390 observations per selected asset;
- at least 57 observed selected assets on every date;
- at least 500 pairwise observations for every selected pair;
- at least 78 fully observed five-replicate weeks;
- at least one naturally missing panel cell and at most 10% total missingness;
- a finite positive balanced-one-way target boundary with between aspect ratio
  below one.

The proof reports the full within-window distributions of week sizes, per-
asset observation counts, per-date observed-asset counts, pairwise overlap,
missingness, between/within degrees of freedom and aspect ratios, complete
balanced weeks, and the independently cross-checked target boundary.

## Predeclared headline calibration claim

The future headline mechanism claim is successful only if every frozen v5
development geometry stratum passes, without pooling rescue:

- at nominal 5%, the exact 95% null interval contains 5% and has upper bound at
  most 7.5%;
- at 1.5 times the independently computed population boundary, FJS-only
  detection and acceptance are each at least 80%, null-to-power gain is at
  least 50 percentage points, and the power curve is nondecreasing;
- squared cosine is at least 0.80, planted-component attribution at least 90%,
  and nuisance attribution at most 10%;
- standardized rescaling, deterministic row order, asset permutation with
  direction map-back, and group-label permutation all pass.

This is a predeclared future claim, not a result. No calibration, confirmation,
or holdout outcome was observed when it was frozen.

## Stop lines

- A failed bounded coverage proof blocks full geometry derivation.
- A passed bounded proof permits only a later separately authorized full input
  derivation; it does not permit detector outcomes or AWS.
- Calendar 2025 remains the single unopened final holdout.
- No v5 outcome, AWS launch, submission, or empirical performance claim exists
  in this lifecycle.
