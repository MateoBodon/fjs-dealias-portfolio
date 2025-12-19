---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Server Environment

- Python: 3.12.3 (local command `python3 --version`). Repo targets Python ≥3.11 (pyproject.toml).
- Dependencies: NumPy, SciPy, pandas, matplotlib (optional), scikit-learn, numba, nasdaq-data-link, wrds, pyyaml, tqdm, pytest, cvxpy, psutil; dev extras include black, ruff, mypy, pytest-xdist/sugar/timeout.
- Recommended setup: `python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]` (Make `setup`).
- Threading: `meta/runtime.py` enforces deterministic thread caps when `EXEC_MODE=deterministic`; throughput mode relaxes BLAS threads. Set OMP/MKL/NUMEXPR vars explicitly for consistency.
- Data mounts: expect WRDS-style CSVs under `data/` (registry-enforced). Do not modify `data/wrds/` or other mounted raw exports.
- Optional tools: matplotlib enables plots; cvxpy required for MV optimisation (otherwise raise/skip). WRDS access via `src/io/wrds_connect.py` may need environment credentials when used.
- Outputs may be large; keep `reports/`, `.cache/`, `experiments/equity_panel/outputs_*` on storage with sufficient space; avoid deleting past RC drops.
