run_name: 20251220_035705_ticket-14_ticket10-fixup
branch: codex/ticket-14-ticket10-fixup
start_sha: 334e86d7ff94aadce6e2c3f86149c198fd9bfdb0
end_sha: 8555ce177125398d5964300a0b3d093d1760f024
start_dirty:
  - docs/CODEX_SPRINT_TICKETS.md
  - docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
commands_executed:
  - python3 -m venv .venv
  - .venv/bin/pip install -e '.[dev]'
  - . .venv/bin/activate && make test-fast
  - . .venv/bin/activate && make run:equity_nested_smoke_tiny
key_artifacts:
  - calibration/nested_edge_delta_thresholds.json (metadata embedded; config hash 69404b24e2352527538e571956b83fc216e10f67139614ec25f409632b1ea48d)
  - experiments/equity_panel/config.nested.smoke.tiny.yaml (max_windows=3)
  - experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/
  - docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/
datasets:
  - data/returns_daily.csv (registry verified prior runs)
notes: initial make test-fast failed (pytest missing) before venv setup; first smoke attempt failed due to NameError on max_windows, fixed by passing max_windows into _run_single_period.
