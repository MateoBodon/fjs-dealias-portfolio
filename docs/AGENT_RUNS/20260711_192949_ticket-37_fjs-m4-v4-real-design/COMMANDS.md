# Commands

Material commands and outcomes, in execution order:

1. Project OS orientation
   - `project-os-v3 status --summary --project-id fjs`
   - `project-os-v3 context --project-id fjs --outcome <bounded-v4-outcome>`
   - Resumed active goal `goal_71e41ee4ddc6`; no new goal or routed worker was
     created.
2. Frozen predecessor and source inspection
   - Verified local v3 `HEAD`/tree
     `b86d98f2a39457920cf9ae4428c2aef517a99e4a` /
     `f2e60230d7df88f637a348e67c994d128e7f6fbc` and the exact remote tree.
   - Verified frozen v2/v3 manifest hashes and the status-ok 2013-2018 CRSP
     receipt chain.
3. Deterministic v4 implementation
   - Added `src/fjs/real_design_contract.py` and
     `tools/freeze_fjs_m4_real_design_v4.py`.
   - Added exact receipt/file binding, frozen filters, PERMNO identity,
     lagged-cap ranking, past-only FF6 regression, deterministic serialization,
     and fail-closed validators.
4. Focused tests
   - `ruff check ...`
   - `python3 -m pytest -q tests/tools/test_fjs_m4_real_design_v4.py`
   - Result: `5 passed`.
5. Bounded actual-source proof
   - Ran the v4 freezer against only
     `month=2013-01/data.csv.gz`, receipt manifest
     `20260707T214900Z_worker8_crsp_dsfv2_month_2017_2010_csvgz`, and the
     registered FF5+MOM factor file.
   - Scanned the complete 141,542-row partition in 25,000-row chunks; no full
     multi-partition derivation ran.
   - A first attempt failed loudly on two date/PERMNO duplicates. Inspection
     proved the required analytical fields were exact while only distribution
     event fields differed. The contract now collapses only exact required-field
     duplicates and rejects any conflict; focused tests cover both paths.
   - Final proof cell and manifest were written outside Git under
     `/tmp/fjs_m4_v4_bounded_proof_20260711/`.
6. Proportional repository and Project OS verification
   - Recorded in `TESTS.md` after completion.

No WRDS login, AWS command, full calibration, detector outcome, headline run,
or 2025 data access occurred.
