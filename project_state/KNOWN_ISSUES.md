---
generated: 2026-07-10T17:32:11-04:00
git_sha: 193a325dc681ebc4da67b44715a92e4f63113019
git_branch: portfolio/fjs-recenter-m1-20260710
commands:
  - Ticket 37 live artifact and code audit
  - sha256 reconciliation of the minimal Ticket 24 detector reference
---
# Known Issues

- **T-012 is recovered but not cleanly ratified**: the four-leg daily DoW matrix artifacts were recovered and appear scientifically usable, but the original T-012 review failed because the long-run monitoring/audit trail was not fully preserved.
- **T-012 does not identify an FJS effect**: all 6,917 changed full-regime windows in the four recovered legs are attributed to the generic `coarse` fallback. Across all eight leg-by-portfolio QLIKE comparisons the overlay loses to the best implemented CC/EWMA comparator, and EW MSE worsens in every leg. The result is historical coarse-overlay evidence, not detector validation.
- **Daily DoW remains empirical-only**: T-008/T-010/T-012 do not recover the clean weekly one-way/FJS theory story. Candidate-source labels are now mandatory, so future arms cannot pool `fjs`, `coarse`, `oracle`, and `sham` candidates.
- **Heavy T-012 details are local-only**: the full recovered tree is under `/Volumes/Storage/Projects/fjs/_recovery/recovered_artifacts/rc-t-012`; Git tracks only curated summary surfaces under `docs/artifacts/rc-t-012/`.
- **Ticket 24 is off-target for between-component power**: `docs/artifacts/detector-contract-reference/ticket24_week_full_fix/curve.csv` shows detection/acceptance at 0.0 across `mu` values 0, 3, 6, 12, and 24, but exact source/command review shows an iid observation-level outer-product injection with no component-mode provenance. It remains exact-config negative-control evidence; the reducer now rejects it for a target `between` claim.
- **The deterministic FJS reference gate is blocked**: `make detector-reference-gate` reports five issues: one-way bulk dimension, inclusion order, explicit-`C_s` MP mapping, spectral reconstruction, and missing target-power treatment provenance. The independent oracle and exact values are frozen in `src/fjs/reference_oracle.py` and `docs/strategy/FJS_REFERENCE_HARNESS_FINDINGS.md`; three strict expected failures keep code mismatches visible.
- **The old headline universe is invalid**: `data/returns_daily.csv` lacks PERMNO/security-filter provenance and the former `assets_top` behavior selected ticker labels alphabetically. The runner now fails closed without an explicit dated ranking snapshot, but a rolling CRSP CIZ/lagged-market-cap adapter is still required for the flagship result.
- **2024/2025 CRSP receipts need dedicated manifests**: the item receipts are successful, but their enclosing manifest has unrelated failures. A content-hashed derived manifest must bind each year before use. Calendar 2025 remains the unopened holdout.
- **Modern baselines remain incomplete**: current `rie` is simple convex shrinkage and current `quest` is MP clipping; neither is authoritative nonlinear shrinkage/QuEST. Robust nonlinear shrinkage, valid POET/SAF, and the large dynamic baseline must fail loudly until independently validated.
- **Advisor-ready headline run is intentionally blocked**: no full CRSP run may start until the exact Ticket 37 detector stop-line passes and at least 30 non-overlapping confirmation dates are changed by the FJS-only arm.
- **Nested design remains secondary**: calibration coverage improved, but tiny real-data nested smoke still has zero detections and skip reasons dominated by stability/no-isolated-spike, so nested is not currently a headline path.
