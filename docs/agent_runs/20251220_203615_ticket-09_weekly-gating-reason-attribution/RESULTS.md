Synthetic diagnostic-failure smoke:
- Command: PYTHONPATH=/root/fjs-dealias-portfolio:/root/fjs-dealias-portfolio/src RUN_NAME=$RUN_NAME python3 docs/agent_runs/$RUN_NAME/synth_diag_failure.py
- Output: experiments/equity_panel/outputs_ticket-09_synth_failure_20251220_203615_ticket-09_weekly-gating-reason-attribution/
- gating_diagnostics.csv: 9/9 windows with skip_reason_primary=diagnostic_failure; exception_type=RuntimeError, exception_stage=dealias_search, exception_message_short populated (<=200 chars). New columns present (skip_reason_primary/detail, exception_type/stage/message_short, replicates).
- weekly_diagnostics.md: counts table shows diagnostic_failure=9 (100%), example windows list fit/hold ranges, p/T/reps, delta, edge, guard tallies, and exception context.

Real deterministic equity smoke:
- Command: EXEC_MODE=deterministic make run:equity_smoke (Makefile now uses python3 and --gating-diagnostics)
- Output dir: experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/
- gating_diagnostics.csv head (window_index, skip_reason_primary, skip_reason_detail, exception_type, exception_stage, exception_message_short):
  - [0, no_isolated_spike, isolated_spikes=0, "", "", ""]
  - [1, "", "", "", "", ""]
  - [2, "", "", "", "", ""]
- Reason counts: no_isolated_spike=1; diagnostic_failure=0; guard_other absent (no guard_other column, no values in diag_payload).
- weekly_diagnostics.md: detection_rate=75%; Primary Skip Reasons table -> no_isolated_spike 1 (100%); guard totals tvec_compute_error=72, tvec_target_zero=2, tvec_off_component=1074; examples section lists window stats with p/T/reps/delta/edge/guards.

Bundle: docs/gpt_bundles/20251220_210439_ticket-09_20251220_203615_ticket-09_weekly-gating-reason-attribution.zip (contents: docs/agent_runs/20251220_203615_ticket-09_weekly-gating-reason-attribution/bundle_contents.txt)
