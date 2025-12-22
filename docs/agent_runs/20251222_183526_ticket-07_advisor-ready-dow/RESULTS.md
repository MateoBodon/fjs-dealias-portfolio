# Results

## Run outputs
- reports/rc-ticket-07-20251222_183800/dow-paper-v1/
- reports/rc-ticket-07-20251222_183800/summary/
  - summary_perf.csv
  - summary_detection.csv
  - overlay_forensics.csv
  - limitations.md
  - advisor_snapshot.md

## Headline eligibility checks
From reports/rc-ticket-07-20251222_183800/dow-paper-v1/run.json (windows block):
```
cap_active: False
cap_sources: []
window_coverage: 1.0
windows_requested: 3512
windows_evaluated: 3512
windows_dropped_holdout_empty: 115
windows_dropped_reasons: {'holdout_empty': 115}
```

Summary artifacts (non-empty):
```
summary_perf.csv: rows=6
summary_detection.csv: rows=3
overlay_forensics.csv: rows=6996
limitations.md: exists, no "run capped" section
```

summary_perf full regime (comparison_valid_* == 1 and n_effective >= 50):
```
EW: comparison_valid_mse/es/qlike/delta=1; n_effective=1749; delta_mse=2.635418515787517e-11; delta_qlike=-0.0671866909475027
MV: comparison_valid_mse/es/qlike/delta=1; n_effective=1749; delta_mse=-6.654496181059978e-13; delta_qlike=-0.0357629174555866
```

Detection + change rates (full regime):
- detection_rate_mean: 0.0416229200503975
- percent_changed (overlay_forensics, full regime, all portfolios): 100.0000%

## Advisor snapshot
- reports/rc-ticket-07-20251222_183800/summary/advisor_snapshot.md

## Data/security checks
- python3 scripts/check_data_policy.py: PASS (check_data_policy: OK)
- Secret scan (rg): hits in docs/CLOUD.md, src/utils/credentials.py, project_state indexes, and agent logs; no secrets committed.
- Restricted-data scan (tracked files): no matches (rg exit 123).
