# Results

- Conflict recorded: AGENTS.md requires a clean git status before starting; user explicitly requested proceeding with a dirty working tree and to keep their local changes uncommitted. Proceeding per user request.

## Code changes
- `tools/make_summary.py`: exclude capped runs (and mv_skip_on_missing_solver runs) from `summary_perf.csv` / `summary_detection.csv`; add limitations sections listing excluded capped runs with cap_sources and smoke-only MV-skip runs; empty summaries now carry headers to avoid downstream errors.
- `experiments/eval/run.py`: ensure `run.json` is written even when `metrics_df` is empty, including `cap_active`, `cap_sources`, and window coverage fields.
- `src/meta/completeness.py`: document run_manifest/run.json as the source of truth for cap/window metadata.
- `tests/tools/test_make_summary.py`: added regression test building a tiny RC directory with one capped + one uncapped design; asserts capped run excluded and limitations list cap_sources.

## RC-lite sanity (deterministic)
- `EXEC_MODE=deterministic make rc-lite-sanity` timed out after 120s, 300s, and 600s while running the vol leg; dow leg completed; vol leg completed manually; weekly legs not run.
- Manual completion: `reports/rc-20251221-sanity-20251221_045550/{dow-tyler,vol-tyler}`.
- Summary run: `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251221-sanity-20251221_045550`.
- Limitations snippet (caps surfaced by date truncation):
  - `reports/rc-20251221-sanity-20251221_045550/summary/limitations.md`
    - `## Excluded smoke-only runs (capped)`
    - `- reports/rc-20251221-sanity-20251221_045550/dow-tyler (cap_sources: date_truncation)`
    - `- reports/rc-20251221-sanity-20251221_045550/vol-tyler (cap_sources: date_truncation, window_coverage)`
- Summary tables are empty (capped exclusions):
  - `reports/rc-20251221-sanity-20251221_045550/summary/summary_perf.csv` rows: 0
  - `reports/rc-20251221-sanity-20251221_045550/summary/summary_detection.csv` rows: 0

## Capped smoke (required)
- Run: `EXEC_MODE=deterministic python -m experiments.eval.run ... --max-windows 5 --out reports/smoke_cap_test`.
- Summary: `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/smoke_cap_test`.
- Limitations snippet:
  - `reports/smoke_cap_test/summary/limitations.md`
    - `## Excluded smoke-only runs (capped)`
    - `- reports/smoke_cap_test (cap_sources: max_windows, window_coverage)`
- Summary tables are empty as expected:
  - `reports/smoke_cap_test/summary/summary_perf.csv` rows: 0
  - `reports/smoke_cap_test/summary/summary_detection.csv` rows: 0
