# Advisor Decision — Daily DoW Robustness Pair (T-010)

This memo evaluates one bounded robustness cycle around the accepted T-008
daily `dow` empirical lane. It does not claim detector validation, theory
recovery, or anything beyond the locked empirical contract.

## Accepted T-008 anchor result

- The accepted T-008 daily `dow` result is `headline-eligible` under the locked
  empirical contract.
- Full-regime T-008 rows:
  - `ew`: `ΔQLIKE=-0.0671866909475027`, `ΔMSE=2.635418515787517e-11`
  - `mv`: `ΔQLIKE=-0.0357629174555866`, `ΔMSE=-6.654496181059978e-13`
- T-008 gate status:
  - `cap_active=False`
  - `window_coverage=1.0`
  - both portfolios present
  - all full-regime `comparison_valid_*` fields are `1.0`
  - `n_effective_qlike=1749.0` for both `ew` and `mv`

## T-010 ff5mom anchor result

- The T-010 `ff5mom` anchor run completes successfully and reproduces the
  accepted T-008 full-regime headline metrics exactly.
- Full-regime T-010 `ff5mom` rows:
  - `ew`: `ΔQLIKE=-0.0671866909475027`, `ΔMSE=2.635418515787517e-11`
  - `mv`: `ΔQLIKE=-0.0357629174555866`, `ΔMSE=-6.654496181059978e-13`
- The T-010 `ff5mom` anchor remains comparison-valid:
  - `cap_active=False`
  - `window_coverage=1.0`
  - all full-regime `comparison_valid_mse`, `comparison_valid_es`,
    `comparison_valid_qlike`, `comparison_valid_dm`, and
    `comparison_valid_delta` fields are `1.0`
  - `n_effective_qlike=1749.0` for both portfolios
  - mandatory baseline rows are present in `full/metrics.csv`

## T-010 no-prewhiten ablation result

- The no-prewhiten ablation also completes successfully under the same daily
  `dow` / `rie` / `tyler` / `126` / `21` / `60` deterministic contract, with
  factor prewhitening disabled as the only intended scientific change.
- Full-regime no-prewhiten rows:
  - `ew`: `ΔQLIKE=-0.3184013657026124`, `ΔMSE=4.8377913766085405e-09`
  - `mv`: `ΔQLIKE=-0.0839030253658935`, `ΔMSE=-6.151598103370174e-12`
- The no-prewhiten run also satisfies the same headline gate:
  - `cap_active=False`
  - `window_coverage=1.0`
  - both portfolios present
  - all full-regime `comparison_valid_*` fields are `1.0`
  - `n_effective_qlike=1796.0` for both `ew` and `mv`
  - mandatory baseline rows are present in `full/metrics.csv`
- Within-run QLIKE versus baseline improves for both portfolios in the
  no-prewhiten ablation.
- The no-prewhiten baseline level is different from the prewhitened baseline,
  so the larger absolute `ΔQLIKE` magnitude is not a claim that the ablation is
  globally better in every sense; it is a within-run baseline comparison under
  the locked contract.

## Headline contract versus diagnostic surfaces

- The active T-010 pass/fail gate remains the locked QLIKE-centered empirical
  contract inherited from T-008: uncapped full-regime rows, full coverage,
  comparison-valid metrics, and QLIKE improvement versus baseline for both
  `ew` and `mv`.
- `reports/rc-t-010/summary/kill_criteria.json` is retained as a stricter
  diagnostic surface, not as the T-010 headline gate. It can fail on EW
  `ΔMSE` while the accepted empirical contract still passes.
- `reports/rc-t-010/summary/completeness.json` currently leaves aggregate
  coverage-count fields as `null`; that is a summary-surface limitation, not
  evidence that the completed T-010 runs are incomplete, because both run
  directories exist and the full-regime rows report `window_coverage=1.0`.

## Decision

- Both T-010 runs remain comparison-valid.
- Both T-010 runs improve QLIKE versus baseline for both `ew` and `mv`.
- The no-prewhiten ablation does not fail the T-008 headline gate and does not
  lose the two-portfolio QLIKE improvement while the anchor passes.
- `n_changed` and `changed_frac` remain fully engaged in both runs:
  - `ff5mom`: `1749` changed windows per portfolio, `changed_frac=1.0`
  - `noprewhiten`: `1796` changed windows per portfolio, `changed_frac=1.0`
- EW `ΔMSE` is positive in both the accepted anchor and the no-prewhiten
  ablation, so this is not a claim of universal loss improvement. It is a
  bounded empirical robustness result under the repo’s current QLIKE-centered
  headline contract.
- Daily `dow` is still not detector-validated.
- Daily `dow` remains an empirical-only lane and does not restore the original
  weekly `oneway` / FJS theory story.

`empirical-lane-robust-enough-to-continue`
