- Removed silent config fallback for explicit `--config` by raising a clear FileNotFoundError when the path is missing.
- `run.json` now records `git_dirty`, `resolved_config_path`, and `resolved_config_hash` (sha256 of `resolved_config.json`).
- Added regression tests for missing config failure + paper-v1 config path resolution, and asserted new run metadata fields.
- Updated docs: removed the “missing paper-v1 config” Known Issue and marked the PLAN_OF_RECORD roadmap item as done.

Artifacts:
- Real-data smoke: `reports/rc-20251223-sanity-20251223_064808/` (daily dow/vol runs + summary).
- Weekly smoke outputs: `experiments/equity_panel/outputs_rc-lite-20251223_20251223_064808/`.
- Sanity summary: `reports/rc-20251223-sanity-20251223_064808/summary_sanity.json`.

Failures: none.
