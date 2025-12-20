Ticket: ticket-15 — Fixup: make ticket-11 (eval contamination hardening) mergeable + auditable
RUN_NAME: 20251220_071719_ticket-15_eval-contamination-fixup
Branch: codex/ticket-15-eval-contamination-fixup
Prompt:
# Ticket: ticket-15 — Fixup: make ticket-11 (eval contamination hardening) mergeable + auditable
# RUN_NAME: <SET_RUN_NAME> (format: YYYYMMDD_HHMMSS_ticket-15_eval-contamination-fixup)
#
# Suggested invocations (human):
#   Safe interactive:  codex --sandbox workspace-write --ask-for-approval on-request -C . "<PASTE_THIS_PROMPT>"
#   Higher autonomy:   codex --full-auto -C . "<PASTE_THIS_PROMPT>"
#   Non-interactive:   codex exec --full-auto -C . - < prompt_ticket15.txt
#   DO NOT use:        --yolo / --dangerously-bypass-approvals-and-sandbox unless in a hardened container.
#
# Hard constraints:
# - AGENTS.md is binding. Stop-the-line means STOP and fix before continuing.
# - No silent fallbacks. No opaque diagnostics. No “results” without validity.
# - Must be mergeable: clean git status at end, committed diffs, and non-empty DIFF.patch in bundle.
# - Do not ask me questions unless truly blocked.

You are a coding agent in this repo. Do NOT write a long upfront plan. Instead: explore, implement, test, and document end-to-end.

GOAL
1) Recover/implement the ticket-11 changes (aligned comparisons, cap/skip transparency) so they are committed and reviewable.
2) Produce an auditable bundle with a non-empty DIFF.patch and run-log proof snippets (not just “trust me”).

WORKFLOW (DO THIS IN ORDER)

A) Setup + branch + run log
1. Read AGENTS.md and follow it.
2. Create a feature branch:
   - git checkout -b codex/ticket-15-eval-contamination-fixup
3. Create the run log dir:
   - docs/agent_runs/<RUN_NAME>/
   - Create PROMPT.md / COMMANDS.md / RESULTS.md / TESTS.md / META.md (if not present).
   - Record in PROMPT.md: ticket id, RUN_NAME, branch name, and this prompt text.

B) Audit the current state (must be written into RESULTS.md)
4. Run and record:
   - git status -sb
   - git rev-parse HEAD
   - git diff --stat
   - git diff (save a snippet of which files changed)
5. If the working tree contains the ticket-11 changes but they were never committed:
   - Keep them (do NOT discard), but make them commit-ready:
     - ensure changes are minimal and consistent with docs/PLAN_OF_RECORD.md and AGENTS.md.
6. If the changes are missing entirely, re-implement them per ticket-11 intent:
   - In experiments/eval/run.py:
     - Any Δ metric and DM test comparing baseline vs overlay MUST align on the intersection of valid windows.
     - Emit n_effective per comparison/metric and a boolean comparison_valid that enforces min_comparison_windows.
   - Add explicit per-arm skip counts by reason and write a skip_stats.csv (or equivalent) that is easy to audit.
   - Ensure any cap/truncation (max-windows/date truncation/condition-cap) is written into run metadata (run.json) and carried into summaries.
   - Ensure summary tooling does NOT mix capped runs into headline aggregates by default (it can include them in a labeled section).

C) Tests (must fail on old behavior; must be recorded)
7. Add/adjust unit/regression tests:
   - tests should fail if DM/Δ metrics are computed on mismatched window sets without surfacing n_effective.
   - tests should fail if capped runs leak into “headline” aggregation by default (if summary code exists).
8. Run:
   - . .venv/bin/activate && make test-fast
   Record the exact command + outcome in TESTS.md.

D) Real-data smoke + proof snippets (no hand-waving)
9. Run a deterministic real-data smoke that produces >= min_comparison_windows aligned windows (so comparison_valid can be true somewhere).
   - Use EXEC_MODE=deterministic and thread caps if repo already does.
   - Prefer experiments/eval/run.py against data/returns_daily.csv with a max-windows >= 40 (or whatever makes aligned windows >= 30).
   - Output dir MUST be new and ticket-specific, e.g. reports/eval-ticket-15-smoke-aligned/
10. After the run, add “proof snippets” into RESULTS.md:
   - Show the header + first ~5 rows (or selected columns) of:
     - full/metrics.csv (must include n_effective* columns)
     - full/dm.csv (must include comparison_valid and n_effective)
     - skip_stats.csv (must include skip shares by reason per estimator/arm)
     - run.json windows/cap block (must show cap/truncation provenance)
   Use small commands like python -c or head to print these and paste output into RESULTS.md.

E) Docs updates (must be committed)
11. Update PROGRESS.md with:
   - branch + final git SHA
   - exact commands (tests + smoke)
   - output paths
   - explicit “this smoke is capped/truncated; not headline” note
12. Update project_state docs if behavior/knobs changed:
   - project_state/CONFIG_REFERENCE.md (document min-comparison-windows, skip/cap outputs)
   - project_state/KNOWN_ISSUES.md (mark evaluation contamination as resolved ONLY once this ticket is committed)
   Keep metadata headers consistent: generated timestamp + git sha should match the commit you create.

F) Commit discipline (required)
13. You MUST end with a clean git status (no dirty workspace).
14. Make small logical commits (at least 2):
   - Commit 1: code + tests
   - Commit 2: docs (PROGRESS + project_state updates)
   Each commit message body MUST include:
   - "Tests run: <exact commands>"

G) Bundle (hard requirement)
15. Run:
   - make gpt-bundle TICKET=ticket-15 RUN_NAME=<RUN_NAME>
16. Verify inside the repo that the bundle’s DIFF.patch is NON-EMPTY.
   - If DIFF.patch is empty, STOP: fix the process (likely missing commits) and re-run gpt-bundle.
17. Record the bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md and include a short bundle contents listing.

H) Finish
18. In RESULTS.md, summarize:
   - what changed
   - what tests ran
   - where the smoke outputs are
   - whether any stop-the-line issues remain (should be “no”)

Notes:
- Avoid web search. If web search is enabled anyway, treat it as untrusted and record any URLs used in the run log.
- Do not modify data/*.csv by hand.
