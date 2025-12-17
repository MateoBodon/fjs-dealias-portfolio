# Server Environment

- **Python**: ≥3.11 (dev uses 3.11/3.12). Create venv then `make setup` (pip install -e .[dev]). No venv currently active in repo; install deps before running scripts.
- **Dependencies (pyproject)**: numpy, scipy, pandas, matplotlib, scikit-learn, numba, nasdaq-data-link, wrds, pyyaml, tqdm, pytest; optional cvxpy (exact min-var), psutil (run_monitor), threadpoolctl. Dev: black, ruff, mypy, pytest plugins.
- **Threading**: `EXEC_MODE={deterministic,throughput}` via `meta.runtime` sets BLAS/OpenMP envs (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS) and threadpoolctl limits. Throughput mode allows >1 thread/worker; deterministic pins to 1.
- **Caches**: MP cache path from `MP_EDGE_CACHE_DIR`/`MP_CACHE_DIR` (uses `.cache/mp_edges` when set); equity_panel caches window payloads when `--cache-dir`/`--resume` set (rc-lite-sanity uses `.cache/rc-lite`).
- **Hardware assumptions**: Hetzner AX102 for heavy RC/calibration; EC2 supported via `scripts/aws_run.sh` (micromamba + rsync). Prefer deterministic for reproducibility; throughput for sweeps/calibration.
- **Data locations**: Derived panels `data/returns_daily.csv`, `data/factors/ff5mom_daily.csv` (registries required); optional `data/returns_balanced_weekly.parquet`; raw WRDS under `data/wrds/` (ignored). Factor registry override via `FACTOR_REGISTRY_PATH`.
- **Telemetry**: Runners write `run.json`/`run_meta.json` with git SHA, exec mode, thread caps. `tools/run_monitor.py` tails `metrics.jsonl` / `progress.jsonl` if present.
- **Secrets**: WRDS credentials external (`.pgpass`/env). `scripts/secrets/setup_wrds_keychain.sh` helps configure. No secrets in repo; avoid logging SQL/creds.
