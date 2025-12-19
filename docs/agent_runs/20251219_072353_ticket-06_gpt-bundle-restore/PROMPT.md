TICKET: ticket-06 (restore make gpt-bundle + enforce review-bundle integrity)

You are Codex running inside Codex CLI in this repo.

Hard rules (stop-the-line):
- Read and follow AGENTS.md first. If AGENTS.md conflicts with this prompt, stop and record the conflict in docs/agent_runs/<RUN_NAME>/RESULTS.md.
- Do NOT claim anything is fixed unless tests pass AND a bundle is generated AND you can point to artifacts in the run log.
- This ticket is infrastructure/process only: do NOT change gating thresholds, solver logic, or research semantics.
- No silent “best effort” bundles: if required inputs are missing, fail loudly (nonzero) with a clear message.

Definition of done (all must be true):
1) `make gpt-bundle TICKET=ticket-06 RUN_NAME=<RUN_NAME>` exists and succeeds on this branch.
2) The produced zip contains the required review payload:
   - AGENTS.md
   - PLAN_OF_RECORD.md
   - DOCS_AND_LOGGING_SYSTEM.md
   - CODEX_SPRINT_TICKETS.md
   - project_state/{CURRENT_RESULTS,KNOWN_ISSUES,CONFIG_REFERENCE}.md
   - PROGRESS.md
   - DIFF.patch (covering this ticket’s code changes)
   - LAST_COMMIT.txt (git log -1 --stat at minimum)
   - docs/agent_runs/<RUN_NAME>/*
3) Add at least one deterministic regression test that would have failed if gpt-bundle was missing/broken.
4) Run and record tests (`make test-fast` minimum).
5) Finish by generating a new bundle (using make) and recording its path + zip file listing in RESULTS.md.

Step 0 — Setup branch + run log
- Create a feature branch: ticket-06-gpt-bundle-restore
- Choose RUN_NAME = YYYYMMDD_HHMMSS_ticket-06_gpt-bundle-restore
- Create docs/agent_runs/<RUN_NAME>/{PROMPT.md,COMMANDS.md,RESULTS.md,TESTS.md,META.json}
  - PROMPT.md must contain this prompt verbatim.
  - COMMANDS.md must append every command you run, in order, including env vars.
- Confirm git status is clean before any “validation” action. If not clean, stop and record why.

Step 1 — Diagnose current state (why Ticket-05 couldn’t bundle)
- Inspect Makefile targets and history:
  - `make -qp | rg -n "^gpt-bundle:"` (or grep equivalent)
  - open Makefile and search for gpt-bundle target
  - check whether docs/DOCS_AND_LOGGING_SYSTEM.md and other required docs exist
- Record findings in docs/agent_runs/<RUN_NAME>/RESULTS.md:
  - Is gpt-bundle target missing? (yes/no)
  - Which required files are missing (if any)

Step 2 — Restore/implement gpt-bundle in Makefile (fail-loud)
- Implement (or restore) a `gpt-bundle:` Makefile target that:
  - Requires TICKET and RUN_NAME variables (fail with clear message if missing)
  - Copies the required files into a temp dir (mktemp -d)
  - Copies docs/agent_runs/<RUN_NAME>/ if it exists (and fail if missing, because a run log is required)
  - Writes DIFF.patch (git diff for this ticket; prefer `git diff main...HEAD` OR an explicit documented range)
  - Writes LAST_COMMIT.txt (git log -1 --stat)
  - Zips the temp dir into: docs/gpt_bundles/<timestamp>_<TICKET>_<RUN_NAME>.zip
  - Prints the absolute path of the zip at the end
- Important: if any “required” source file is missing (e.g., docs/DOCS_AND_LOGGING_SYSTEM.md), do NOT silently continue. Fail loudly.

Step 3 — Restore missing required docs if they were accidentally deleted
- If docs/DOCS_AND_LOGGING_SYSTEM.md is missing:
  - Restore it from git history if possible (preferred: `git log -- docs/DOCS_AND_LOGGING_SYSTEM.md` then `git show <sha>:docs/DOCS_AND_LOGGING_SYSTEM.md > ...`)
  - If history does not contain it, recreate a minimal version consistent with current repo conventions (docs/agent_runs, commit-body test recording, etc.)
- Do the same for docs/PLAN_OF_RECORD.md and docs/CODEX_SPRINT_TICKETS.md if missing.
- Do NOT fabricate project_state docs content; if missing, stop and record it as a new blocker.

Step 4 — Repo hygiene: prevent committing run outputs
- Check if reports/ or bundle outputs are tracked in git:
  - `git status --porcelain`
  - `git ls-files | rg -n "^(reports/|bundles/|docs/gpt_bundles/)"` (or grep)
- If run outputs are tracked, remove them from git (keep on disk) and add proper .gitignore entries.
  - Keep docs/agent_runs/<RUN_NAME>/ tracked (run logs are allowed/required).
  - Do NOT commit large report directories as part of this ticket.

Step 5 — Tests
- Add one regression test that would catch this class of failure:
  Example options (pick one, keep it small):
  A) A pytest that runs `make -qp` and asserts the `gpt-bundle:` target exists.
  B) A small python tool (tools/verify_gpt_bundle.py) + pytest that checks a created zip contains required paths.
- Run minimum test suite:
  - `make test-fast`
  - any targeted pytest you added
- Record exact commands + outcomes in:
  - docs/agent_runs/<RUN_NAME>/TESTS.md
  - commit message bodies

Step 6 — Validate gpt-bundle end-to-end
- Run:
  - `make gpt-bundle TICKET=ticket-06 RUN_NAME=<RUN_NAME>`
- Then verify bundle contents:
  - `unzip -l docs/gpt_bundles/*ticket-06*<RUN_NAME>*.zip | tee docs/agent_runs/<RUN_NAME>/bundle_contents.txt`
- Record in RESULTS.md:
  - the bundle path
  - confirmation that required files are present
  - any missing files (should be none)

Step 7 — Commits + documentation
- Keep commits small and logical:
  1) Makefile target restore/implementation
  2) restore missing docs (if needed)
  3) tests
  4) .gitignore / hygiene
- Every commit MUST include in the commit body:
  - Tests: <exact commands>
  - Artifacts: <bundle path and/or relevant outputs>
- Update PROGRESS.md with:
  - timestamp, sha, RUN_NAME, bundle path, what changed

Finish
- Ensure `make gpt-bundle ...` succeeds and bundle contents are verified and logged.
- Ensure working tree is clean at end.
- Generate the bundle one final time and record the final bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md.
