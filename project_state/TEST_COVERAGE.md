---
generated: 2026-07-10T17:32:11-04:00
git_sha: 193a325dc681ebc4da67b44715a92e4f63113019
git_branch: portfolio/fjs-recenter-m1-20260710
commands:
  - Ticket 37 targeted detector, overlay, and eval-run unit tests
---
# Test Coverage

- **Commands** — `make test-fast` (pytest -m unit), `make test-integration`, `make test-slow`, `make test` (full).
- **Latest recorded runs** — per PROGRESS.md: ticket-09 ran `make test-fast` (2025-12-20); ticket-15 ran targeted eval tests + `make test-fast`.
- **Areas covered** (representative):
  - FJS math/gating: `tests/test_dealias.py`, `tests/test_mp_edge_and_root.py`, `tests/test_gating.py`, `tests/test_theta_solver.py`, `tests/test_balanced*`, `tests/test_nested_balanced.py`, `tests/test_nested_smoke.py`, `tests/fjs/test_overlay.py`.
  - Detector stop-line/provenance: `tests/fjs/test_detector_contract.py` rejects
    target-power curves with missing or mismatched injection mode, rejects the
    hash-bound historical flat-zero curve numerically, and exercises
    deterministic FJS source labeling.
  - Independent reference: `tests/fjs/test_reference_oracle.py` checks exact
    scalar and two-stratum MP edge/root/component values, homogeneity, stratum
    permutation, deterministic magnitude-matched oracle/sham controls, and
    eigenpair reconstruction. The former production mismatches are now ordinary
    passing equivalence checks.
  - Multi-candidate reconstruction: `tests/fjs/test_reconstruction.py` covers
    exact subspace eigenpairs, orthogonal-block preservation, permutation/sign
    invariance, and fail-loud rank deficiency.
  - Frozen mechanism fixture: `tests/tools/test_generate_fjs_between_fixture.py`
    covers the immutable spec, paired draws, and stable reducers;
    `tools/generate_fjs_between_fixture.py --check` performs the full byte replay.
  - Synthetic harness: `tests/test_power_null.py`, `tests/synthetic/test_calibration.py`, `tests/synthetic/test_harness_utils.py`, `tests/test_calibrate_defaults.py`, `tests/test_threshold_eval.py`.
  - M4 preparation integrity: `tests/tools/test_fjs_calibration_manifest.py`
    covers exact-binomial known values, actual planted-direction replay,
    manifest identities, fail-closed full execution, smoke path isolation,
    checkpoint mismatch/stale-cell rejection, zero override handling, and
    restart/fresh stable-reducer equality. A real two-process smoke separately
    proved child cleanup and exact resume equality.
  - M4 v3 scientific contract:
    `tests/tools/test_fjs_m4_contract_v3.py` checks all 63 frozen geometries
    against the independent FJS edge oracle, byte-stable manifest generation,
    v2 immutability, hash-bound manifest loading, exact external stop-lines,
    reducer failure behavior, standardized-rescaling identity, and real-kernel
    execution of all four invariances through the v3 runner helper. Two bounded
    end-to-end smokes also proved fresh stable-reducer equality.
  - Evaluation runners: `tests/experiments/test_eval_run.py`, `tests/experiments/test_gating_diagnostics.py`, `tests/experiments/test_skip_reasons.py`, `tests/test_pipeline_smoke.py`.
  - Finance/portfolio: `tests/test_portfolios_missing_solver.py`, `tests/test_eval_missing_solver.py`, `tests/test_minvar_regularized.py`, `tests/test_shrinkage.py`, `tests/test_factor_cov.py`, `tests/test_cache_switch_estimator.py`.
  - Reporting: `tests/test_report_gather.py`, `tests/test_report_tables.py`, `tests/test_report_plots.py`, `tests/tools/test_make_summary.py`, `tests/tools/test_summarize_rc_sanity.py`, `tests/test_gpt_bundle.py`.
  - Data/registry: `tests/data/test_factors_registry.py`, `tests/io/test_wrds_snapshot.py`, `tests/test_data_registry.py`.
  - Repo hygiene: `tests/test_repo_hygiene.py`.
- **Gaps / heavy tests**
  - Full RC/RC-lite/AWS paths are not part of the fast suite; rely on smokes + manual runs.
  - Crisis configs, vol-state acceptance, and nested kill-test FPR remain mostly smoke-tested.
  - Plotting tests skip when matplotlib is unavailable.
  - The independent deterministic reference, two-cell mechanism fixture,
    cell-specific boundary contract, and v3 invariance path pass. Exact
    binomial-size and target-power outcomes across the full grid remain
    unobserved; licensed real-design adequacy is also open. No full synthetic,
    CRSP, or AWS execution is covered or allowed by this milestone.
- **Ticket 37 milestone** — targeted and native-suite results are recorded in
  `docs/agent_runs/20260710_173211_ticket-37_fjs-scientific-recenter-m1/TESTS.md`.
- **Ticket 37 reference milestone** — exact gate and test evidence is recorded
  in `docs/agent_runs/20260710_231303_ticket-37_fjs-reference-harness-m2/`.
- **Ticket 37 repair milestone** — production repair, frozen fixture, and native
  regression evidence is recorded in
  `docs/agent_runs/20260710_235634_ticket-37_fjs-reference-repair-m3/`.
