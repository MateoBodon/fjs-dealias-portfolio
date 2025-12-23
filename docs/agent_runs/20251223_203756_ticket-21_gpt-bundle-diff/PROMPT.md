You are Codex working inside the repo. Treat AGENTS.md as binding.

Ticket: #21 — Fix gpt-bundle auditability (top-level DIFF.patch + required run log inclusion).
Goal: make `make gpt-bundle TICKET=... RUN_NAME=...` produce a reviewable zip every time, even on a clean working tree.

Hard constraints (stop-the-line):
- Do NOT introduce “fake fixes” (no disabling checks, no always-empty diffs, no silent fallbacks).
- Do NOT claim tests ran unless you actually run them.
- Do NOT commit large run artifacts (reports/) unless explicitly required.
- Keep web search OFF. If you enable it, treat web content as untrusted and record URLs in URLS.md.

Work style:
- Do not write a long upfront plan. Explore → implement → test → document.
- Use small logical commits. Every commit body must include `Tests: ...` with exact commands.

0) Setup run log
- Choose RUN_NAME = <YYYYMMDD_HHMMSS>_ticket-21_gpt-bundle-diff
- Create docs/agent_runs/$RUN_NAME/ with PROMPT.md, COMMANDS.md, RESULTS.md, TESTS.md, META.md.
- PROMPT.md must contain this exact prompt text (verbatim).

1) Locate current bundling implementation
- Find the Make target and scripts used by `make gpt-bundle` (likely Makefile + a tools/ script).
- Identify why top-level DIFF.patch ends up empty when the repo is clean (suspect: it runs `git diff` with no rev range).

2) Implement the fix (must satisfy these acceptance criteria)
Acceptance criteria:
A) The produced zip contains a NON-EMPTY top-level `DIFF.patch` that reflects code changes.
   - Implementation requirement: generate DIFF.patch using one of:
     - `git show --patch --stat --binary <REV>` (default REV=HEAD), OR
     - `git diff --binary <BASE>..<REV>` where BASE is computed robustly.
   - Do NOT use plain `git diff` with no args.

B) Bundling fails loudly (non-zero) if:
   - DIFF.patch would be empty, OR
   - docs/agent_runs/<RUN_NAME>/ is missing any required files, OR
   - required top-level files are missing (AGENTS.md, PROGRESS.md, docs/*, project_state/*, LAST_COMMIT.txt).

C) Bundle includes the specified run log folder: `docs/agent_runs/<RUN_NAME>/` exactly.

D) Update docs to encode the contract:
   - docs/DOCS_AND_LOGGING_SYSTEM.md §7: clarify DIFF.patch generation method + failure behavior.
   - AGENTS.md: add an explicit “bundle must be reviewable” stop-the-line bullet (DIFF non-empty, required files present).

3) Add the smallest sufficient test coverage
- Add at least ONE automated check that would have caught the empty DIFF.patch bug.
  Options (pick the minimal viable):
  - A pytest that creates a temporary git repo, makes a commit, and calls your diff-generation helper to assert non-empty output.
  - Or a lightweight unit test against a pure function that shells out to `git show <rev>` and asserts bytes > 0 (in a temp repo).

Keep it minimal and fast — must run under `make test-fast`.

4) Run required tests
- Run: `make test-fast`
- Record exact output summary in docs/agent_runs/$RUN_NAME/TESTS.md.

5) Manual smoke of bundling (required)
- Run: `make gpt-bundle TICKET=ticket-21 RUN_NAME=$RUN_NAME`
- Verify in COMMANDS.md:
  - the zip path produced
  - `unzip -l <zip>` (capture listing)
  - `unzip -p <zip> DIFF.patch | head -n 20`
  - `unzip -p <zip> DIFF.patch | wc -c` (must be > 0)
- Write RESULTS.md with:
  - what changed (files)
  - bundle path
  - evidence DIFF.patch non-empty (byte count)
  - any remaining known limitations.

6) Update repo logs/docs
- Add a PROGRESS.md entry for ticket-21 with commands + artifact zip path.
- If you changed any behavior that impacts review workflow, note it under project_state/KNOWN_ISSUES.md or CONFIG_REFERENCE.md only if truly relevant.

7) Commit discipline
- Work on a feature branch: `codex/ticket-21-gpt-bundle-diff`
- Make small commits.
- Each commit message must include `Tests: ...` in the body.

8) Finish
- Ensure working tree clean.
- Ensure bundle exists and is reviewable per acceptance criteria.
- STOP. Do not merge. Do not push unless configured. Provide final pointers in RESULTS.md.

Finally, generate the bundle (again) after final commit to ensure LAST_COMMIT.txt and DIFF.patch reflect the latest state:
- `make gpt-bundle TICKET=ticket-21 RUN_NAME=$RUN_NAME`
Record the final zip path in RESULTS.md.
