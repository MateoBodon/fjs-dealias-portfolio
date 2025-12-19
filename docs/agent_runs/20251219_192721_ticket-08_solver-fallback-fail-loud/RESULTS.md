# RESULTS

- cvxpy present in environment (pip install -e .[dev] reports satisfied requirement).
- Deterministic smoke: `EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --out reports/eval-smoke-ticket08 --max-windows 2 --assets-top 50 --overlay-delta 0.2 --mv-box-lo 0.0 --mv-box-hi 0.1` — succeeded; MV weights computed via internal ridge solver (no fallback needed); solver dependency present.
- Bundle: `docs/gpt_bundles/20251219_194020_ticket-08_20251219_192721_ticket-08_solver-fallback-fail-loud.zip`.
