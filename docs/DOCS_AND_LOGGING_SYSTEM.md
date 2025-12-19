# Docs and Logging System

Scope
- Document how work is recorded and bundled for review.
- Applies to sprint logs under docs/agent_runs/, project_state/*.md, and PROGRESS.md.

Run logs (docs/agent_runs/<RUN_NAME>/)
- Required files: PROMPT.md (verbatim ticket), COMMANDS.md (every command in order), RESULTS.md (findings/outcomes), TESTS.md (commands + results for tests), META.json (run metadata).
- Create a fresh RUN_NAME per sprint: YYYYMMDD_HHMMSS_<ticket>_<short-description>.
- Never reuse or overwrite a prior run_dir; append instead.

Recording rules
- Append commands immediately to COMMANDS.md with the exact invocation (including env vars).
- Capture test invocations + pass/fail in TESTS.md.
- Summarize state changes and blockers in RESULTS.md; include bundle paths when generated.
- Do not delete prior run logs or reports without explicit approval.

Bundles
- gpt-bundle packages AGENTS.md, PLAN_OF_RECORD.md, DOCS_AND_LOGGING_SYSTEM.md, CODEX_SPRINT_TICKETS.md, project_state/*.md, PROGRESS.md, the current run_dir, DIFF.patch, and LAST_COMMIT.txt.
- Required inputs must exist; bundling should fail loudly if any are missing.
- Bundles are written to docs/gpt_bundles/<timestamp>_<TICKET>_<RUN_NAME>.zip and the absolute path must be printed.

Change management
- When updating documentation rules or bundle contents, update this file and PROGRESS.md and note the RUN_NAME.
- Keep entries ASCII where possible; include timestamps in ISO or YYYYMMDD_HHMMSS.
