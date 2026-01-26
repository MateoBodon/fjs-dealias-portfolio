# Prompt

You are in the FJS repo root. Goal: enforce TRACKING_POLICY.md conventions and eliminate dirty trees caused by run output directories and local-only ignores.

Hard rules:
- Do NOT commit unrelated existing work; only normalization/migration changes.
- Do NOT keep shared policy rules in .git/info/exclude.
- Run dumps must go under reports/_runs/ (ignored) or artifacts/_local/ (ignored).
- Tickets are tracked under docs/tickets/. Agent run logs are tracked under docs/agent_runs/ (small text/json only).

Steps:

1) Baseline:
   - git status --porcelain
   - cat .git/info/exclude

2) Ensure scaffold exists:
   - TRACKING_POLICY.md exists (already added).
   - Ensure dirs exist: docs/tickets docs/agent_runs docs/artifacts reports/_runs artifacts/_local data/samples data/schema configs/local .cache tmp
   - Ensure README placeholders exist in docs/tickets, docs/agent_runs, docs/artifacts.

3) Enforce .gitignore:
   - Confirm .gitignore includes canonical scratch zones + docs/gpt_outputs/docs/local/docs/prompts and does NOT ignore docs/agent_runs.
   - Confirm .env.example and .env.local.example are not ignored.
   If missing, edit .gitignore.

4) Fix .git/info/exclude drift:
   - If .git/info/exclude contains policy rules (docs/gpt_outputs, docs/prompts, reports/rc-..., etc), migrate them into .gitignore and clear .git/info/exclude.

5) Clean run dumps that cause dirty trees:
   - For each untracked directory under reports/ that looks like a run dump (timestamped, many files, smoke runs, etc):
     - Move it into reports/_runs/<topic>/<original_dirname>/ preserving structure.
     - Ensure it is ignored and disappears from git status.
   - Example: reports/inject_spike_smoke/... should become reports/_runs/inject_spike_smoke/...

6) Track tickets and curated run logs:
   - Any docs/tickets/*.md should be added and committed.
   - Any docs/agent_runs/** should be added and committed (but only small structured run logs; do not add huge binaries).

7) Stage only normalization changes:
   - git add TRACKING_POLICY.md .gitignore docs/tickets/README.md docs/agent_runs/README.md docs/artifacts/README.md
   - git add docs/tickets/*.md (new/updated)
   - git add docs/agent_runs/** (small logs only)
   Do NOT stage unrelated code changes.

8) Commit:
   - "chore: normalize tracking policy and move run dumps to reports/_runs"

9) Verify clean:
   - git status --porcelain should not show untracked run dumps anymore.

10) History audit (plan only):
   - git count-objects -vH
   - top blobs list
   - If big accidental blobs exist, write docs/history_cleanup_plan.md (do not run filter-repo).

Deliverables:
- One commit that removes the dirty-tree sources (run dumps) and normalizes tracking
- Clean working tree (or clearly explained remaining unrelated changes)
- Optional history cleanup plan
