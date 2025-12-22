---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Server Environment

- OS: `Linux Ubuntu-2404-noble-amd64-base 6.8.0-85-generic #85-Ubuntu SMP PREEMPT_DYNAMIC Thu Sep 18 15:26:59 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux`
- Python: `Python 3.12.3` (repo targets Python ≥3.11 in `pyproject.toml`).
- Dependencies (from `pyproject.toml`): numpy, scipy, pandas, matplotlib (optional), scikit-learn, numba, nasdaq-data-link, wrds, pyyaml, tqdm, pytest, cvxpy, psutil; dev extras include black, ruff, mypy, pytest-xdist/sugar/timeout.
- Recommended setup: `python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]` (or `make setup`).
- Threading: `meta/runtime.py` enforces deterministic thread caps when `EXEC_MODE=deterministic`; throughput mode relaxes BLAS threads. Set OMP/MKL/NUMEXPR vars explicitly for reproducibility.
- Data mounts: expect WRDS-style CSVs under `data/` (registry-enforced). Do not modify raw WRDS exports if present.
- Optional tools: matplotlib enables plots; cvxpy required for MV optimisation (otherwise fail-loud unless skip flag set).
