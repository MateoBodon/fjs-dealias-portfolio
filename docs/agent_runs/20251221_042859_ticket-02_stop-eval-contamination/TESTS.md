# Tests

- `make test-fast` (pass; 69 passed, 162 deselected, 1 warning: PytestConfigWarning unknown timeout option).
- `EXEC_MODE=deterministic make rc-lite-sanity` (timed out at 120s, 300s, 600s; dow leg completed; vol leg completed manually; weekly legs not run).
- `EXEC_MODE=deterministic PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py ... --group-design vol --out reports/rc-20251221-sanity-20251221_045550/vol-tyler` (pass).
- `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251221-sanity-20251221_045550` (pass).
- `EXEC_MODE=deterministic python -m experiments.eval.run ... --max-windows 5 --out reports/smoke_cap_test` (pass).
- `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/smoke_cap_test` (pass).
