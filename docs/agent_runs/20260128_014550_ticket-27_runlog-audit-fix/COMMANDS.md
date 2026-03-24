# Commands

- python3 tools/agentic/runlog_init.py --ticket "27" --summary "Ticket-27 run log audit fixes" --run-name "20260128_014550_ticket-27_runlog-audit-fix"
- unzip -p artifacts/_local/gpt_bundles/20260127_162507_27_20260127_053650_ticket-27_repo-hygiene-cleanup.zip BUNDLE_META.md
- (edit) docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/META.md
- (create) docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/META.json
- (edit) docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/RESULTS.md
- (edit) docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/TESTS.md
- (edit) PROGRESS.md
- . .venv/bin/activate && make validate-runlogs
- . .venv/bin/activate && make test-fast
- make gpt-bundle TICKET=ticket-27 RUN_NAME=20260128_014550_ticket-27_runlog-audit-fix
