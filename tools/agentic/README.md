# tools/agentic

Deterministic helper scripts used by the Agentic System Kit.

- `repo_snapshot.py` → creates `docs/_generated/repo_snapshot.md`
- `project_state_refresh.py` → updates `project_state/_generated/*` and can create `project_state.zip`
- `gpt_bundle.py` → creates `gpt_bundle.zip` for GPT review
- `runlog_init.py` → creates `docs/agent_runs/<RUN_NAME>/` with standard log files (`META.json` canonical, `META.md` legacy)
- `validate_runlog.py` → validates required run log files (single run or `--all`; legacy `META.md` accepted unless strict)
- `ticket_new.py` → creates a ticket template under `docs/tickets/`
