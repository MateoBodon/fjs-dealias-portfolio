PASS (diagnostic loop completed; week curve remains flat-zero with tvec root/off-component dominance).

Changes:
- Reclassified t-vector failures into explicit guard reasons (no real root / no admissible root / singularity) to avoid lumping into tvec_compute_error.
- Extended inject_spike diagnostics to track new guard keys and tvec-dominance logic.
- Added debug-window fixture + unit test for tvec no-root behavior.

Week inject-spike full run (post-fix):
- Run id: 20251226_ticket24_week_full_fix
- Outputs: reports/inject_spike/20251226_ticket24_week_full_fix/
- Curve:
  - mu=0.0 detect=0.00 accept=0.00 (n=186)
  - mu=3.0 detect=0.00 accept=0.00 (n=74)
  - mu=6.0 detect=0.00 accept=0.00 (n=74)
  - mu=12.0 detect=0.00 accept=0.00 (n=74)
  - mu=24.0 detect=0.00 accept=0.00 (n=74)
- Dominant pre-gate reasons (mu=0): tvec_off_component=22320, tvec_no_real_root=7756, tvec_no_admissible_root=3404; tvec_compute_error=0.
- Profile: mp.t_vec/admissible_m_from_lambda dominates (see artifacts/profile_week_full_fix.txt).

Artifacts copied:
- docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/curve_week_full_fix.csv
- docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/gating_reasons_week_full_fix.csv
- docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/windows_detail_week_full_fix.csv
- docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/profile_week_full_fix.txt
- docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/debug_window_week_full_fix.npz
- docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/curve_week_full_fix.png

Bundle:
- docs/gpt_bundles/20251226_081511_ticket-24_20251226_060917_ticket-24_finish-week-inject-spike.zip

Tests: python -m pytest tests/experiments/test_inject_spike.py -q; make test-fast
