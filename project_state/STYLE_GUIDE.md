# Style Guide (observed + recommended)

- **Language**: Python 3.11+; type hints on public functions; prefer short, composable helpers over dense one-liners.
- **Formatting/Lint**: `ruff` + `black` + `mypy`; abide by `pytest.ini` markers; `_`-prefixed helpers for internal use; avoid non-ASCII unless already present.
- **Numerical care**: symmetrise covariance matrices (`0.5*(A+A.T)`), clip eigenvalues, guard against NaNs/inf, use `np.asarray(..., dtype=np.float64)`, ridge when needed.
- **Config handling**: deep-merge YAML/JSON/CLI (`experiments.eval.config`), validate inputs early with clear errors, prefer deterministic seeds.
- **Logging/telemetry**: record run metadata (`run_meta.json`), cache keys include code signature; avoid printing data/credentials; keep console output concise.
- **Plotting**: use `matplotlib.use("Agg")` in non-interactive contexts; create parent dirs before saving; close figures promptly.
- **Testing expectations**: run `make test-fast` before commits; mark heavier tests with `slow`/`heavy` to keep CI lean.
- **Git/branching**: feature branches `codex/<task>`; commits prefixed `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `perf:` per AGENTS.
- **Documentation**: update PROGRESS.md after RC/calibration; keep memos/briefs reproducible; prefer apply_patch for edits.
