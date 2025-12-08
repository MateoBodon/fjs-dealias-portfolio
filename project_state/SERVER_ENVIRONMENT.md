# Server Environment

- **Python**: >=3.11 (tested on 3.11/3.12). Recommended: `python -m venv .venv && source .venv/bin/activate` then `make setup` (editable install + dev extras) and `pre-commit install` if config present.
- **Dependencies (pyproject)**: numpy, scipy, pandas, matplotlib, scikit-learn, numba, nasdaq-data-link, wrds, pyyaml, tqdm, pytest; optional: cvxpy (min-var), psutil. Dev: black, ruff, mypy, pytest-*.
- **Threading**: `EXEC_MODE={deterministic,throughput}` via `meta.runtime` sets BLAS/OpenMP envs (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS) and threadpoolctl if available. Throughput mode scales worker count down and allows 2–4 threads per worker.
- **Caches**: MP edge cache dir from `MP_EDGE_CACHE_DIR`/`MP_CACHE_DIR` (created if set); per-window stats cached under `.cache/` by equity_panel runner when `--cache-dir`/`--resume` used.
- **Hardware assumptions**: Hetzner AX102 (reference) for heavy RC/calibration; EC2 supported via `scripts/aws_run.sh` (micromamba + rsync). Use `EXEC_MODE=throughput` for long sweeps on Hetzner; deterministic for reproducible metrics.
- **Data locations**: committed derived panels under `data/returns_daily.csv`, `data/factors/ff5mom_daily.csv` with registries; raw WRDS exports under `data/wrds/` (ignored). Factor registry path can be overridden with `FACTOR_REGISTRY_PATH`.
- **Telemetry**: Runners write `run.json`/`run_meta.json` with git SHA, exec mode, thread caps. `tools/run_monitor.py` tails `metrics.jsonl` / `progress.jsonl` if produced.
- **Secrets**: WRDS credentials external (`.pgpass` or env); `scripts/secrets/setup_wrds_keychain.sh` helps configure. No secrets in repo; do not log SQL/credentials.
