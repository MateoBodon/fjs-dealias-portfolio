start_sha: 312f02c5c4eb9500ca62bf5254f9959f2c482871
end_sha: 312f02c5c4eb9500ca62bf5254f9959f2c482871
branch: codex/ticket-11-eval-contamination
dirty: true
runs:
  - reports/eval-ticket-11-smoke-small/
commands:
  - make test-fast
  - EXEC_MODE=deterministic PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py --returns-csv data/returns_daily.csv --window 40 --horizon 5 --max-windows 4 --group-min-replicates 2 --assets-top 30 --prewhiten off --use-factor-prewhiten 0 --out reports/eval-ticket-11-smoke-small
