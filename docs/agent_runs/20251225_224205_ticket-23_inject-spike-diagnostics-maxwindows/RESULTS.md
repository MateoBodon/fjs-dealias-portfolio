# Results — ticket-23

## Summary of changes
- Added deterministic max-windows sampling and per-window diagnostics in `experiments/eval/inject_spike.py`.
- `detect_spikes` now records diagnostics + gating reason buckets; `inject_spike` emits `windows_detail.csv` and `gating_reasons.csv` plus run.json summaries.
- Unit tests added for deterministic sampling, new output schemas, and missing-config hard errors.

## Key outputs
- DoW inject spike run (completed): `reports/inject_spike/20251225_ticket23_dow_tyler/`
  - Copied to run log: `docs/agent_runs/20251225_224205_ticket-23_inject-spike-diagnostics-maxwindows/artifacts/curve_dow_tyler.csv`
  - Copied to run log: `docs/agent_runs/20251225_224205_ticket-23_inject-spike-diagnostics-maxwindows/artifacts/gating_reasons_dow_tyler.csv`
- Gating attribution (DoW, μ=0 baseline) dominated by:
  - `tvec_off_component=22380`
  - `tvec_compute_error=10211`

## DoW curve (20251225_ticket23_dow_tyler)
| mu | detection_rate | acceptance_rate | n_windows | n_detected | n_accepted |
|---:|---------------:|----------------:|----------:|-----------:|-----------:|
| 0.0 | 0.00 | 0.00 | 25 | 0 | 0 |
| 3.0 | 0.00 | 0.00 | 1 | 0 | 0 |
| 6.0 | 0.00 | 0.00 | 1 | 0 | 0 |
| 12.0 | 0.00 | 0.00 | 1 | 0 | 0 |
| 24.0 | 0.00 | 0.00 | 1 | 0 | 0 |

## Week run status
- Multiple week/scm/coarse runs were attempted but aborted locally due to long `dealias_search` runtime; no completed week curve.
- Aborted runs left only `resolved_config.json` under their report directories (not used for results).

## Acceptance criteria
- **FAIL** — No design+edge_mode showed detection/acceptance increasing with μ; week run could not be completed locally.

## Bundle + tests
- Bundle: `docs/gpt_bundles/20251226_004354_ticket-23_20251225_224205_ticket-23_inject-spike-diagnostics-maxwindows.zip`
- Tests: `make test-fast`
