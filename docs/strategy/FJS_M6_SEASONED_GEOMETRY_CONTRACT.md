# FJS M6 Seasoned-Universe Geometry Contract

status: frozen before any M6 geometry result
as_of: 2026-07-11
active_goal: `goal_fbb8a01af913`

M5 remains an unchanged real negative: its 2013-01 point-in-time top-60 panel
passed 9 of 11 gates but missed the frozen 500-pairwise and 78-complete-week
requirements. M6 is a distinct additive method change, not a threshold rescue.
Every M5 coverage and future detector-outcome gate remains unchanged.

## Eligibility derived from the frozen requirements

Let `N` be the number of factor-calendar dates in the 156-week window and let
the already-frozen minimum pairwise overlap be `L = 500`. If each eligible
asset has at least

`ceil((N + L) / 2)`

observations, then any pair has at least `2 * ceil((N + L) / 2) - N >= L`
overlapping observations by inclusion-exclusion. This is a sufficient rule
derived only from `N` and the frozen 500 gate.

Separately, select the most recent 78 factor-calendar weeks containing all five
weekday dates before formation. An eligible asset must be observed on every
date in this common deterministic anchor set. Any selected top-60 panel then
has at least 78 fully observed five-replicate weeks by construction.

The rule does not use M5's observed values of 438 pairwise dates or 72 complete
weeks. Those values motivated a method change but do not set any M6 constant.

## Unchanged boundaries

- Same exact 2010-2012 warm-up, 2013-01 bounded endpoint, 156-week window,
  CIZ filters, point-in-time PERMNO identity, cap ranking, and tie-break.
- Same top-60 size and every M5 coverage threshold.
- Same predeclared all-strata null, 1.5-times-boundary planted-power,
  attribution, and invariance claim.
- Source return presence and numeric validity are read for eligibility; no
  return values, detector outputs, or performance summaries are persisted.
- Detailed PERMNO/member/mask proof stays outside Git. Only compact aggregate
  manifest/readback may be published.
- One bounded 2013-01 proof only. No full corpus, detector outcome, AWS, or
  calendar 2025 access.
