# Campaign Decision — Daily DoW Full Matrix (T-012)

This memo evaluates the frozen four-leg daily `dow` empirical campaign under the accepted empirical-only lane. It does not claim detector validation, theory recovery, or anything beyond the locked daily `dow` contract.

## W126 control reproduction

- Both `w126` control legs reproduce the ratified T-010 full-regime truth exactly for both `ew` and `mv` on all contract fields checked: `delta_qlike_vs_baseline`, `delta_mse_vs_baseline`, `cap_active`, `window_coverage`, all `comparison_valid_*` fields, `n_effective_qlike`, `n_changed`, and `changed_frac`.
- T-012 `ff5mom_w126` full rows:
  - `ew`: `ΔQLIKE=-0.0671866909475027`, `ΔMSE=2.635418515787517e-11`
  - `mv`: `ΔQLIKE=-0.0357629174555866`, `ΔMSE=-6.654496181059978e-13`
- T-012 `noprewhiten_w126` full rows:
  - `ew`: `ΔQLIKE=-0.3184013657026124`, `ΔMSE=4.8377913766085405e-09`
  - `mv`: `ΔQLIKE=-0.0839030253658935`, `ΔMSE=-6.151598103370174e-12`

## W252 robustness result

- `ff5mom_w252` remains uncapped, full-coverage, and comparison-valid for both portfolios, with mandatory baseline rows present and `run.json` reporting `status=ok`, `stage=complete`.
  - `ew`: `ΔQLIKE=-0.0095291564243905`, `ΔMSE=1.3216196016995945e-11`, `n_effective_qlike=1688.0`, `n_changed=1688`, `changed_frac=1.0`
  - `mv`: `ΔQLIKE=-0.0054396781873731`, `ΔMSE=3.6290396942467586e-14`, `n_effective_qlike=1688.0`, `n_changed=1688`, `changed_frac=1.0`
- `noprewhiten_w252` remains uncapped, full-coverage, and comparison-valid for both portfolios, with mandatory baseline rows present and `run.json` reporting `status=ok`, `stage=complete`.
  - `ew`: `ΔQLIKE=-0.0327990066509463`, `ΔMSE=3.120412943940577e-09`, `n_effective_qlike=1666.0`, `n_changed=1666`, `changed_frac=1.0`
  - `mv`: `ΔQLIKE=-0.0092719760222629`, `ΔMSE=9.639339228606278e-13`, `n_effective_qlike=1666.0`, `n_changed=1666`, `changed_frac=1.0`
- Both `w252` legs still improve QLIKE versus baseline for both `ew` and `mv`, but the improvement magnitude is materially smaller than the corresponding `w126` controls, especially in the prewhitened leg.

## Claim Boundary

- Daily `dow` is still not detector-validated.
- Daily `dow` remains an empirical-only lane and does not restore the original weekly `oneway` / FJS theory story.
- `reports/rc-t-012/summary/kill_criteria.json` remains a stricter diagnostic surface, not the pass/fail gate for this campaign. It still fails the EW `ΔMSE <= 0` check even though the locked empirical gate passes.
- `reports/rc-t-012/summary/completeness.json` still leaves aggregate coverage-count fields as `null`; that remains a summary-surface limitation rather than missing-run evidence because all four run directories are complete and the full-regime rows report `window_coverage=1.0`.

## Decision

- The two `w126` control legs reproduce the ratified T-010 truth exactly.
- At least one `w252` leg remains uncapped, comparison-valid, and QLIKE-improving for both `ew` and `mv`; in fact, both `w252` legs satisfy that gate.
- The longer-window axis weakens the headline deltas materially, but it does not collapse the current daily `dow` empirical lane under the locked contract.

empirical-lane-still-worth-scaling
