# Commands

Executed from repo root unless noted.

```bash
# Context and uploaded source discovery
git status --short
find docs -maxdepth 2 -type f | rg -i 'analysis\\.md$|analysis'
wc -l docs/Analysis.md
sed -n '1,120p' docs/Analysis.md

# Run-log initialization
python3 tools/agentic/runlog_init.py --ticket "33" --summary "Ingest uploaded Analysis.md into canonical full review artifact and refresh bundle" --run-name "20260216_220117_ticket-33_ingest-uploaded-analysis-and-rebundle"
git status --porcelain > docs/agent_runs/20260216_220117_ticket-33_ingest-uploaded-analysis-and-rebundle/STATUS_START.txt

# Canonical artifact + ticket metadata updates
cp docs/Analysis.md docs/gpt_outputs/20260216_project_review_full.md

# Required checks
. .venv/bin/activate && make validate-runlogs
. .venv/bin/activate && make test-fast

# Bundle generation (post-commit)
. .venv/bin/activate && BUNDLE_STAMP=20260216_220359 make gpt-bundle TICKET=33 RUN_NAME=20260216_220117_ticket-33_ingest-uploaded-analysis-and-rebundle
unzip -p artifacts/_local/gpt_bundles/20260216_220359_33_20260216_220117_ticket-33_ingest-uploaded-analysis-and-rebundle.zip BUNDLE_META.md
```

File edits were applied to:
- `docs/Analysis.md` (uploaded source tracked for provenance)
- `docs/gpt_outputs/20260216_project_review_full.md`
- `docs/CODEX_SPRINT_TICKETS.md`
- `docs/tickets/ticket-33_canonical_project_review_and_codex_prompt.md`
- `docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/META.json`
- `docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/META.md`
- `docs/agent_runs/20260216_220117_ticket-33_ingest-uploaded-analysis-and-rebundle/*`
- `PROGRESS.md`
