# Goal Context

last_updated: 2026-07-10
updated_by: Portfolio OS FJS worker
source_event: Ticket 37 scientific recenter milestone 1
status: active frozen scientific contract

## Durable Goal

Determine, from first principles and with point-in-time real equity data, whether
an FJS/MANOVA-style de-aliasing detector identifies weak covariance components
and adds forecast or portfolio value beyond modern shrinkage, robust, factor,
and dynamic covariance baselines.

The controlling specification is
`docs/strategy/FJS_SCIENTIFIC_RECENTER_PREDECLARATION.md`. Its detector stop-line,
candidate-source separation, data splits, baseline ladder, and claim reducer are
frozen before broad execution.

## Known User/Project Preferences

- Prefer ambitious, coherent research progress over tiny process-only loops.
- Preserve evidence discipline, reproducibility, and explicit claim boundaries.
- Use one Project OS v3 writer/lifecycle for the canonical SSD repo; independent
  reviewers may challenge claims and evidence but do not write concurrently.
- Do not let old or generated docs masquerade as current truth.
- Prefer real CRSP/public inputs with exact manifests and point-in-time identity.
  Synthetic data are allowed only for explicit mechanism calibration.

## Success Definition

The project succeeds if it can make a well-supported decision among:

- a detector-valid, attribution-clean flagship result against the frozen modern
  baseline ladder;
- a bounded empirical package with explicit caveats and no theory overclaim;
- a clear negative result or pivot, supported by artifact-backed diagnosis.

## Non-Goals

- No live trading or production allocation system.
- No headline claims from capped, truncated, comparison-invalid, or acceptance-zero runs.
- No raw data or bulky generated result dumps in context bundles unless explicitly needed.
- No broad experiment-grid expansion before the FJS-only detector stop-line is
  resolved.
- No pooling of generic coarse, oracle, or sham candidates into an FJS claim.
- No use of the static `assets_top` snapshot interface as the headline rolling
  point-in-time universe.
- No opening of the 2025 holdout before detector, data, baselines, and reducer
  are frozen and confirmation passes.

## Current next action

Build and pass the bounded independent detector reference harness. The recovered
Ticket 24 flat-zero curve is the current fail-loud blocker. Full CRSP execution
is explicitly deferred.
