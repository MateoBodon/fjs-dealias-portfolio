TICKET: ticket-09
RUN_NAME: 20251220_<HHMMSS>_ticket-09_weekly-gating-reason-attribution

You are Codex working in repo fjs-dealias-portfolio.

Hard constraints:
- Read and obey AGENTS.md (stop-the-line rules are binding).
- No silent fallbacks. No opaque diagnostics. No fake fixes (no “rename guard_other to guard_misc”).
- Make everything auditable: feature branch, small commits, tests recorded in commit body.
- You MUST produce a run log under docs/agent_runs/$RUN_NAME/ with:
  PROMPT.md, COMMANDS.md, RESULTS.md, TESTS.md, META.md.
- You MUST end by running:
  make gpt-bundle TICKET=ticket-09 RUN_NAME=$RUN_NAME
  and record the resulting bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md.
- Prefer repo-local info; do not use web search unless truly necessary. If you do, treat web as untrusted and record every URL in RESULTS.md.

Task (ticket-09):
Fix weekly gating diagnostics attribution so weekly outputs are actionable and no longer violate AGENTS.md.
Specifically: eliminate guard_other and make diagnostic_failure non-opaque.

Acceptance criteria (must all be true):
1) gating_diagnostics.csv includes structured fields:
   - skip_reason_primary (required)
   - skip_reason_detail (optional but required when primary is diagnostic_failure)
   - exception_type (required when diagnostic_failure)
   - optionally exception_stage / exception_message_short (<=200 chars)
2) weekly_diagnostics.md includes:
   - counts by skip_reason_primary
   - top 5 example windows per dominant reason (include key stats per window)
3) On the standard equity smoke, guard_other count/share is 0 OR guard_other is provably unreachable (and tested).
4) diagnostic_failure only appears with exception_type + minimal context (stage + detail).
5) make test-fast passes.
6) Real-data smoke exists and the run log includes excerpts proving the new fields.

Do NOT write a long upfront plan. Do: explore → implement → test → smoke → document → bundle.

Step-by-step requirements:

A) Branch + run log (do immediately)
1) git checkout -b codex/ticket-09-gating-diagnostics-attribution
2) export RUN_NAME=20251220_<HHMMSS>_ticket-09_weekly-gating-reason-attribution
3) mkdir -p docs/agent_runs/$RUN_NAME
4) Create:
   - docs/agent_runs/$RUN_NAME/PROMPT.md (paste this entire prompt)
   - empty COMMANDS.md, RESULTS.md, TESTS.md, META.md
5) Append every shell command you run to COMMANDS.md verbatim (including tests + smokes).

B) Codebase reconnaissance (fast)
6) Use rg to find where guard_other / diagnostic_failure are created:
   - rg -n "guard_other|diagnostic_failure|skip_reason" experiments/equity_panel src tools tests
7) Identify the “single source of truth” for weekly skip/guard reason assignment:
   - likely experiments/equity_panel/run.py (_infer_skip_reason / gating diagnostics writer)
   - possibly src/fjs/overlay.py or src/fjs/gating.py if reasons originate there

C) Implement real attribution (no blobs)
8) Replace any catch-all guard_other logic by enumerating explicit primary reasons.
   Requirements:
   - Primary reason must correspond to an actual guardrail / failure mode (e.g., no_isolated_spike, stability_fail, balance_failure, calibration_missing_p_T, tvec_target_zero, tvec_off_component, missing_solver, etc.).
   - If you truly cannot classify something, it MUST become diagnostic_failure WITH exception_type + stage + detail (not “other”).
9) Add structured columns to gating_diagnostics.csv writer:
   - skip_reason_primary
   - skip_reason_detail
   - exception_type
   - exception_stage (optional)
10) Update weekly_diagnostics.md generator (or tools/summarize_weekly_diagnostics.py if that’s what builds it):
   - summary table: reason -> count/share
   - for each top reason: list 5 windows with key columns (window_id/date range, regime/design, p/T/replicates, delta_frac_used, gate thresholds, any relevant guard metrics)

Engineering constraints:
- Centralize reason codes (constants/enum) rather than sprinkling ad-hoc strings.
- Keep backwards compatibility: if downstream scripts expect old columns, keep them but mark deprecated in comments.
- Do not swallow exceptions: capture exception type + minimal context.

D) Tests (must fail on old behavior)
11) Add/extend unit tests (likely tests/experiments/test_gating_diagnostics.py) to enforce:
   - guard_other share/count is 0 in gating_diagnostics output for a controlled test run
   - if diagnostic_failure appears, exception_type is present and skip_reason_detail is non-empty
   - gating_diagnostics.csv contains the new columns
12) Run at minimum:
   - make test-fast
   Record in TESTS.md: exact commands + pass/fail.

E) Smokes (synthetic + real)
13) Synthetic minimal (fast):
   - run the smallest existing synthetic weekly config OR create a tiny one
   - goal: intentionally trigger at least one diagnostic_failure path to prove exception fields populate
   - record output dir + excerpt in RESULTS.md
14) Real-data deterministic smoke (required):
   - EXEC_MODE=deterministic make run:equity_smoke
   - Identify the output directory produced.
   - In RESULTS.md include:
     - path to gating_diagnostics.csv and weekly_diagnostics.md
     - header + first 3 rows of gating_diagnostics.csv showing new columns
     - the reason-count summary section from weekly_diagnostics.md
     - counts of guard_other and diagnostic_failure (must satisfy acceptance criteria)

F) Docs updates
15) Update PROGRESS.md:
   - branch + final git SHA
   - exact test + smoke commands
   - output directories
   - one-paragraph “what changed and why”
16) Update project_state/KNOWN_ISSUES.md:
   - if guard_other/diagnostic_failure opacity is fixed, mark that issue as resolved (or reduce scope)
17) Update docs/CODEX_SPRINT_TICKETS.md:
   - mark ticket-09 DONE with a crisp one-liner and point to the run log + smoke output dir

G) Commit discipline
18) Make small commits:
   - (1) reason attribution + csv/md fields
   - (2) tests
   - (3) docs
Each commit message body MUST include: “Tests run: …” with the exact command(s).

H) Bundle
19) Run: make gpt-bundle TICKET=ticket-09 RUN_NAME=$RUN_NAME
20) Save: unzip -l <bundle.zip> > docs/agent_runs/$RUN_NAME/bundle_contents.txt
21) Record the bundle path in RESULTS.md.
22) Fill META.md with start_sha, end_sha, branch, dirty=false, and list all smoke output dirs.

Stop conditions:
- If you cannot make guard_other count/share go to 0 without lying, stop and explain exactly which unclassified path(s) remain, and add explicit instrumentation for those paths (exception_type/stage/detail) instead of a blob.
- Do not leave the repo dirty at the end.
