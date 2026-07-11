# FJS M4 v4 Full-Generation Finalizer Contract

status: frozen before the 72-month derivation
as_of: 2026-07-11
active_goal: `goal_71e41ee4ddc6`

## Scope

This contract finalizes realistic-design **inputs only**. It cannot run the FJS
detector, inspect calibration outcomes, open 2025, authorize AWS, promote a
model, or make an empirical claim.

The exact required generation is one primary source and one restart identity
for every calendar month from `2013-01` through `2018-12` inclusive:

- required months: 72;
- cell identity: `fjs-real-design-YYYY-MM-v4`;
- security identity inside each cell: PERMNO;
- primary source identity: `month=YYYY-MM` plus its status-ok receipt and full
  content SHA-256;
- factor identity: one exact registered FF5+MOM binding shared by all cells.

Additional prior-month sources used for past-only factor fitting are allowed
only when they resolve exactly inside the same 72-month source catalog.

## Restart protocol

`tools/finalize_fjs_m4_real_design_v4.py` exposes five deterministic actions:

1. `init` creates an empty generation checkpoint with all 72 expected months
   and code/predecessor bindings.
2. `register` independently revalidates a cell, its complete file hash, every
   source receipt/hash, its factor binding, and its month-specific identity.
3. `status` reports exact completed and missing months without reading or
   creating outcomes.
4. `finalize` refuses to proceed until all 72 unique month/cell identities are
   present, then creates the aggregate input manifest.
5. `readback` independently rehashes the 72 source files, 72 cell artifacts,
   factor file, and aggregate manifest and emits a separate readback receipt.

Registering the exact same month receipt after a restart is idempotent.
Registering a different artifact, source, identity, or generation for an
already completed month is a hard error. Checkpoint writes use atomic replace.

## Completeness and integrity gates

Finalization fails on any of the following:

- missing or duplicate month;
- missing, duplicate, or misordered cell identity;
- duplicate artifact path satisfying multiple months;
- source outside 2013-2018 or source absent from the final catalog;
- source receipt, size, content hash, or cell hash drift;
- inconsistent factor binding;
- cross-generation receipt;
- aggregate source-set, cell-set, or manifest digest mismatch;
- any non-false outcome, promotion, AWS, legacy-ticker, or 2025 boundary.

The final manifest can mark the 72-month realistic-design input generation
complete only after all gates pass. It must still report
`full_execution_ready=false`, `aws_execution_authorized=false`, and
`outcomes_present=false`. Detector execution remains blocked on the trusted
route and fresh AWS authority plus a separate explicit run action.

## Frozen predecessor

- Published v4 local tree:
  `040ffb6c8407d9bf7b8b887dc611e37948fb437d`.
- Published v4 remote commit:
  `967fcf5b2c0db171972a85d575149859ddb2ad05`.
- Bounded proof cell file SHA-256:
  `855d3a57673d2bee0ae40c06eb96268ee64dffd6039d6207da52c99d8896d208`.
- Bounded proof manifest file SHA-256:
  `e2d2d9880dc9c5e4533085ce2b396ea6aa152043bed8aa82172c46d4865a3f39`.

No full input manifest or readback receipt exists at contract freeze time.
