# Results

## Additive v4 contract

The v4 implementation turns the former realistic-design placeholder into a
deterministic, independently checkable input-freezing path:

- every CRSP partition is matched to exactly one `status=ok` receipt and bound
  to its complete file SHA-256;
- the source contract fixes CIZ common-equity/security/trading filters and a
  $5 minimum absolute price;
- security identity is PERMNO, with no ticker deduplication or future backfill;
- the universe is ranked by the last eligible lagged `dlycap` observation,
  before the calibration window, with PERMNO as the deterministic tie-break;
- FF5+MOM coefficients are fit only on dates before the window and applied to
  later observations;
- the cell serializes exact universe membership, factor coefficients,
  missingness, week/weekday geometry, aspect ratios, pairwise counts, and PSD
  residual covariance with independent hashes;
- source, cell, and manifest mutation are rejected on readback.

## Bounded real-source proof

- Source partition: `month=2013-01`.
- Source SHA-256:
  `21bd0e46eacc37a8c33ab953da84935163b860481b3083b3f5e28c7cc7524167`.
- Receipt manifest SHA-256:
  `ea993595018676fcc53926a6f801b1ac6184b4ba6864b2175f3e09617918f4bf`.
- Rows receipted/scanned: `141542` / `141542`.
- Rows after frozen filters and exact duplicate collapse: `62572`.
- Exact analytical duplicate rows collapsed: `2`; conflicting duplicates
  accepted: `0`.
- Proof cell: 8 PERMNOs selected at 2013-01-15, 10 past-only FF6 fit dates per
  asset, and 11 subsequent January dates.
- Cell digest:
  `86defc680ab260565a12a7413acc3306779eded536a91a33c25b0bad04e18855`.
- Cell file SHA-256:
  `855d3a57673d2bee0ae40c06eb96268ee64dffd6039d6207da52c99d8896d208`.
- Proof manifest file SHA-256:
  `e2d2d9880dc9c5e4533085ce2b396ea6aa152043bed8aa82172c46d4865a3f39`.

## Honest boundary

The bounded proof establishes input-freezer correctness on one complete source
partition. It does not close the full 72-partition generation, run detector
size/power, produce empirical performance, or authorize promotion/AWS. The
legacy ticker CSV was not used or provenance-inferred. Calendar 2025 remained
unopened. The full v4 derivation may begin only as a separate hash-readback
generation when the portfolio memory-heavy lane is available.
