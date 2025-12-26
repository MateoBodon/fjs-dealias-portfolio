- Verified prior ticket-25 run log completeness by filling missing TESTS.md in 20251226_095630_ticket-25_week-between-smoke.
- Ran fixture smokes for inject_mode within and total (week design, 20 windows, μ={0,12,24}).

Run summaries:
- Within run dir: reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/
- Total run dir: reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/
- Artifacts copied: docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/artifacts/within/ and .../total/

Acceptance check (within):
- curve.csv includes μ=0.0, 12.0, 24.0; detection/acceptance are 0.00 at all μ (n_windows=20 baseline; n_windows=1 at μ=12/24).
- windows_detail.csv μ=24 raw_outliers_found>0 share = 0.0 (0/1).
- Dominant pre-gate reasons at μ=0 and μ=24: tvec_off_component, tvec_no_real_root, tvec_no_admissible_root.

Acceptance check (total):
- curve.csv includes μ=0.0, 12.0, 24.0; detection/acceptance are 0.00 at all μ (n_windows=20 baseline; n_windows=1 at μ=12/24).
- windows_detail.csv μ=24 raw_outliers_found>0 share = 0.0 (0/1).
- Dominant pre-gate reasons at μ=0 and μ=24: tvec_off_component, tvec_no_real_root, tvec_no_admissible_root.

Bundle:
- docs/gpt_bundles/20251226_103353_ticket-25_20251226_102602_ticket-25_week-within-total-smoke.zip
