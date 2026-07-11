# Commands

1. Launched `tools/run_fjs_m4_real_design_full_v4.py` against the frozen local
   2013-2018 CRSP partitions with generation
   `fjs-m4-v4-real-design-2013-2018-g1`.
2. Repaired universe eligibility after the exact fail-closed April 2014 error,
   added regression coverage, and independently regenerated the first 15 cell
   cores for equality.
3. Published the repaired tree, migrated the exact 15-cell checkpoint with
   full artifact readback, and resumed the same generation.
4. Finalized all 72 cells and independently replayed the final manifest
   readback byte-for-byte.
5. Reduced source totals, missingness, degrees of freedom, turnover, and
   lineage counts into `docs/artifacts/fjs-m4-v4-full-inputs/aggregate_receipt.json`.

All generated data stayed under `/Volumes/Storage/Projects/fjs/_runs/`; no AWS,
detector outcome, 2025 input, or credentialed WRDS session was used.
