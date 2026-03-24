# Commands

Executed from repo root unless noted.

```bash
# Baseline context
 git status --short
 git rev-parse HEAD
 ls -1 docs/agent_runs | tail -n 20
 git branch --show-current
 git branch --list 'codex/ticket-33*'

# Branch + runlog init
 git switch -c codex/ticket-33-canonical-project-review
 date -u +%Y%m%d_%H%M%S
 python3 tools/agentic/runlog_init.py --ticket "33" --summary "Canonical project review + Codex continuation prompt + ticket-32 audit drift fix" --run-name "20260216_212107_ticket-33_canonical-review-prompt-audit-fix"
 git status --porcelain > docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/STATUS_START.txt

# Ticket-32 audit drift inspection
 sed -n '1,220p' docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/META.json
 sed -n '1,260p' docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/RESULTS.md
 rg -n "ticket-32|050959|head_sha|bundle generation|errata" PROGRESS.md
 sed -n '520,660p' PROGRESS.md
 ls -1 artifacts/_local/gpt_bundles | rg '_32_'
 unzip -p artifacts/_local/gpt_bundles/20260216_050959_32_20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta.zip BUNDLE_META.md
 unzip -p artifacts/_local/gpt_bundles/20260216_051120_32_20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta.zip BUNDLE_META.md

# Canonical docs + source checks
 sed -n '1,260p' AGENTS.md
 sed -n '1,260p' docs/DOCS_AND_LOGGING_SYSTEM.md
 sed -n '1,260p' docs/PLAN_OF_RECORD.md
 sed -n '1,260p' docs/CODEX_SPRINT_TICKETS.md
 sed -n '1,260p' project_state/CURRENT_RESULTS.md
 sed -n '1,260p' project_state/KNOWN_ISSUES.md
 ls -1 docs/gpt_outputs
 sed -n '1,260p' docs/gpt_outputs/20260216_analysis_full.md
 sed -n '1,360p' docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/PROMPT.md
 find /home/codex -maxdepth 6 -type f -name 'Analysis.md' 2>/dev/null

# Ticket-33 edits and new docs
 cat > docs/prompts/codex_continuation.md <<'EOF_INNER'
 ...
EOF_INNER
 cat > docs/gpt_outputs/20260216_project_review_full.md <<'EOF_INNER'
 ...
EOF_INNER
 cat > docs/tickets/ticket-33_canonical_project_review_and_codex_prompt.md <<'EOF_INNER'
 ...
EOF_INNER
 cat > docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/PROMPT.md <<'EOF_INNER'
 ...
EOF_INNER

# Required validations/tests
 . .venv/bin/activate && make validate-runlogs
 . .venv/bin/activate && make test-fast

# Bundle generation + verification
 . .venv/bin/activate && BUNDLE_BASE=7f7ebd64379bf85d09f968c14b2e68bd9bd43db2 BUNDLE_STAMP=20260216_223500 make gpt-bundle TICKET=33 RUN_NAME=20260216_212107_ticket-33_canonical-review-prompt-audit-fix
 unzip -p artifacts/_local/gpt_bundles/20260216_223500_33_20260216_212107_ticket-33_canonical-review-prompt-audit-fix.zip BUNDLE_META.md
 unzip -p artifacts/_local/gpt_bundles/20260216_223500_33_20260216_212107_ticket-33_canonical-review-prompt-audit-fix.zip DIFF.patch | rg -n "20260216_project_review_full|codex_continuation|ticket-33_canonical_project_review_and_codex_prompt|ticket-32 bundle audit errata"
```

File edits were applied to:
- `PROGRESS.md`
- `docs/CODEX_SPRINT_TICKETS.md`
- `docs/PLAN_OF_RECORD.md`
- `docs/prompts/codex_continuation.md`
- `docs/gpt_outputs/20260216_project_review_full.md`
- `docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/META.json`
- `docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/META.md`
- `docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/RESULTS.md`
- `docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/COMMANDS.md`
- `docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/*`
- `docs/tickets/ticket-31_docs_recenter_snapshot_refresh.md`
- `docs/tickets/ticket-33_canonical_project_review_and_codex_prompt.md`
