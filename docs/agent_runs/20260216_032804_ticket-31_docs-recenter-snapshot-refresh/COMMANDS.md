# Commands

Executed from repo root unless noted.

```bash
date +%Y%m%d_%H%M%S
git status --short
rg --files | rg '^(PROJECT\.md|README\.md|PROGRESS\.md|docs/PLAN_OF_RECORD\.md|docs/CODEX_SPRINT_TICKETS\.md|project_state/CURRENT_RESULTS\.md|tools/agentic/runlog_init\.py|docs/DOCS_AND_LOGGING_SYSTEM\.md)$'

python3 tools/agentic/runlog_init.py --ticket "31" --summary "Docs recenter + snapshot refresh" --run-name "20260216_032804_ticket-31_docs-recenter-snapshot-refresh"

# Artifact checks used to verify CURRENT_RESULTS claims
sed -n '1,40p' reports/rc-ticket-07-20251222_183800/summary/summary_detection.csv
sed -n '1,80p' reports/rc-ticket-07-20251222_183800/summary/summary_perf.csv
sed -n '1,40p' reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv
sed -n '1,80p' reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv
sed -n '1,220p' reports/rc-20251121/metrics_summary.json
sed -n '1,260p' reports/inject_spike/20251226_ticket24_week_full_fix/run.json
sed -n '1,120p' reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv
sed -n '1,80p' reports/inject_spike/20251226_ticket24_week_full_fix/gating_reasons.csv
sed -n '1,240p' reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/run.json
sed -n '1,220p' experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/weekly_diagnostics.md

# Required validations
. .venv/bin/activate && make validate-runlogs
. .venv/bin/activate && make test-fast

# Bundle
. .venv/bin/activate && make gpt-bundle TICKET=31 RUN_NAME=20260216_032804_ticket-31_docs-recenter-snapshot-refresh
unzip -l artifacts/_local/gpt_bundles/20260216_034848_31_20260216_032804_ticket-31_docs-recenter-snapshot-refresh.zip
unzip -p artifacts/_local/gpt_bundles/20260216_034848_31_20260216_032804_ticket-31_docs-recenter-snapshot-refresh.zip DIFF.patch | wc -c
```

File edits were applied to:
- `PROJECT.md`
- `README.md`
- `docs/PLAN_OF_RECORD.md`
- `docs/CODEX_SPRINT_TICKETS.md`
- `docs/gpt_outputs/20260216_analysis.md`
- `project_state/CURRENT_RESULTS.md`
- `project_state/KNOWN_ISSUES.md`
- `project_state/OPEN_QUESTIONS.md`
- `PROGRESS.md`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/*`
