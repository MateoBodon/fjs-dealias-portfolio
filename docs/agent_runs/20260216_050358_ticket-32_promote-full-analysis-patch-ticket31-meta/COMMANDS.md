# Commands

Executed from repo root unless noted.

```bash
git status --short
git show --no-patch --pretty=fuller 8bd1282541112293a3e6c823b7e32bbeaa8ef5c2
git show --no-patch --pretty=%P 8bd1282541112293a3e6c823b7e32bbeaa8ef5c2

python3 tools/agentic/runlog_init.py --ticket "32" --summary "Promote full analysis + patch ticket-31 audit metadata" --run-name "20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta"
git status --porcelain > docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/git_status_start.txt

# Ticket-31 metadata + PROGRESS/PLAN checks
cat docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json
cat docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.md
rg -n "20260216_032804_ticket-31_docs-recenter-snapshot-refresh|1371b3c2e7|8bd1282541112293a3e6c823b7e32bbeaa8ef5c2" PROGRESS.md docs/PLAN_OF_RECORD.md

# Locate analysis source
ls -la docs/gpt_outputs
find /home/codex -type f \( -name 'Analysis.md' -o -name 'analysis.md' -o -name '*analysis_full*.md' \)

# Required validations
. .venv/bin/activate && make validate-runlogs
. .venv/bin/activate && make test-fast
```

File edits were applied to:
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.md`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/git_status_start.txt`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/git_status_end.txt`
- `docs/gpt_outputs/20260216_analysis_full.md`
- `docs/PLAN_OF_RECORD.md`
- `PROGRESS.md`
- `docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/*`
