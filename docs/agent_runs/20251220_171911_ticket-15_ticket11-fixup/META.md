start_sha: b932a0d6ace045508f372afb76284e0c04f03b1a
end_sha: 09bd1eef7e66736af26bb417afece6d7ad993e68
branch: codex/ticket-15-ticket11-fixup
dirty: false
runs:
  - reports/ticket-15-smoke-171911/
commands:
  - pytest tests/experiments/test_eval_run.py::test_aligned_delta_and_dm_use_window_intersection tests/experiments/test_eval_run.py::test_run_evaluation_marks_comparison_valid_and_caps tests/experiments/test_eval_run.py::test_run_evaluation_delta_respects_changed_window_filter
  - make test-fast
  - EXEC_MODE=deterministic python3 -m experiments.eval.run --returns-csv data/returns_daily.csv --window 40 --horizon 5 --out reports/ticket-15-smoke-171911 --assets-top 20 --shrinker rie --use-factor-prewhiten 0 --prewhiten off --q-max 2 --mv-box-lo -0.25 --mv-box-hi 0.25 --mv-turnover-bps 0.0 --mv-condition-cap 1000000 --max-windows 5 --min-comparison-windows 3 --seed 123 --workers 1
