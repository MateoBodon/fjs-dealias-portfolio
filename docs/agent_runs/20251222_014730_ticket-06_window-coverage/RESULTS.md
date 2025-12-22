# Results

## Pre-fix reproduction (ticket-05 evidence)
- `reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json` windows block:
  - `cap_active: true`, `cap_sources: ['window_coverage']`, `window_coverage: 0.9682933553901296`, `windows_requested: 3627`, `windows_evaluated: 3512`.
- `reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv`:
  - `reason_code == holdout_empty` count = 115.
  - `window_id` NaN count = 115; `window_start` NaN count = 115 (all holdout-empty rows missing identifiers).

## Fix verification (ticket-06 real-data run)
Run root: `reports/rc-ticket-06-20251222_063304/`

- `reports/rc-ticket-06-20251222_063304/dow-paper-v1/run.json` windows block:
  - `windows_candidate: 3627`, `windows_after_caps: 3627`.
  - `windows_dropped_holdout_empty: 115`, `windows_dropped_reasons: {'holdout_empty': 115}`.
  - `windows_requested: 3512`, `windows_evaluated: 3512`, `window_coverage: 1.0`.
  - `cap_active: false`, `cap_sources: []`.
- `reports/rc-ticket-06-20251222_063304/dow-paper-v1/diagnostics_detail.csv`:
  - `holdout_empty` rows now carry identifiers (`window_id`/`window_start` non-NaN); `drop_reason` column present.
- Summary outputs are non-empty:
  - `reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv` rows = 6.
  - `reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv` rows = 3.
  - `reports/rc-ticket-06-20251222_063304/summary/overlay_forensics.csv` rows = 6996.
- Validity checks:
  - `comparison_valid_* == 1` for all rows in `summary_perf.csv`.
  - `n_effective` min = 715 (>= 50).
- Limitations:
  - `reports/rc-ticket-06-20251222_063304/summary/limitations.md` includes “windows dropped from planning (holdout_empty: 115)” and no capped-run exclusion section.

## Artifacts
- Run outputs: `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`
- Summary directory: `reports/rc-ticket-06-20251222_063304/summary/`
- GPT bundle: `docs/gpt_bundles/20251222_090028_ticket-06_20251222_014730_ticket-06_window-coverage.zip`
