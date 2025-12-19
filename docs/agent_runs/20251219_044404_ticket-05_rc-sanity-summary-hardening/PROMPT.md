<environment_context>
  <cwd>/root/fjs-dealias-portfolio</cwd>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>bash</shell>
</environment_context>

TICKET: ticket-05 (rc-lite-sanity summary hardening: partial run detection + missing sections)

You are Codex running inside Codex CLI in this repo.

Hard rules:
- Read and follow AGENTS.md (stop-the-line rules are binding).
- Follow docs/DOCS_AND_LOGGING_SYSTEM.md exactly (run logs + commit body requirements).
- This ticket is ONLY about summary/discovery/completeness logic (no gating/threshold/solver changes).
- Do NOT “fix” completeness by silently skipping missing sections or changing defaults. Missing outputs must be surfaced loudly.

Definition of done (must satisfy all):
- Incomplete/partial RC dirs are flagged as incomplete and excluded from aggregates.
- Summary outputs include all expected sections when present; when absent, show explicit “missing/empty” markers.
- Tests pass (make test-fast minimum) and are recorded in commit bodies + docs/agent_runs/<RUN_NAME>/TESTS.md.
- Deterministic rc-lite-sanity validation run exists and is referenced in RESULTS.md.
- Finish by generating a new bundle:
  make gpt-bundle TICKET=ticket-05 RUN_NAME=<RUN_NAME>
  and record the bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md.

Step 0 — Setup run log + branch
1) Create feature branch: ticket-05-rc-sanity-summary-hardening
2) Choose RUN_NAME = YYYYMMDD_HHMMSS_ticket-05_rc-sanity-summary-hardening
3) Create docs/agent_runs/<RUN_NAME>/{PROMPT.md,COMMANDS.md,RESULTS.md,TESTS.md,META.json}
   - PROMPT.md must be this prompt text.
   - COMMANDS.md must append commands you actually run (chronological; include env vars).
4) Confirm git status is clean before “validating runs”. If not clean, stop and note why in RESULTS.md.

Step 1 — Locate current summary/discovery behavior
- Use rg to find where rc-lite-sanity output dirs are discovered and summarized:
  rg -n "summarize_rc_sanity|summary_sanity|rc-lite-sanity|run_manifest|cap_active|window_coverage|incomplete" tools src tests
- Identify:
  - What files are considered “required” for a run to be considered complete
  - Whether missing files currently cause: crash, silent skip, or misleading “valid”
  - Whether daily vol-state and weekly sections are conditionally omitted

Step 2 — Implement completeness-aware summary (no silent skips)
Implement a clear completeness model and wire it into summarize tools:
A) Define a single source of truth for “run completeness”
   - Prefer putting shared logic in src/meta/run_meta.py (or a new src/meta/completeness.py if cleaner).
   - For each run directory being summarized, compute:
     - is_complete: bool
     - missing_files: list[str]
     - incomplete_reason: str (human readable)
     - excluded_from_aggregate: bool (true if incomplete OR cap_active OR window_coverage<1)
   - Required files should at minimum include: run_manifest.json (or run.json mirror), metrics.csv, diagnostics.csv (or whatever the repo’s canonical outputs are).
   - This must also detect “partial RC dir” patterns (e.g., resolved_config/prewhiten-only).

B) Update tools/summarize_rc_sanity.py and tools/make_summary.py to:
   - Always emit a summary_sanity.json that includes:
     - per-run completeness metadata
     - aggregate metrics computed only on complete, uncapped, full-coverage runs
   - Always include the canonical section headers/blocks:
     - daily DoW
     - daily vol-state
     - weekly DoW
     - weekly nested
     If a section is missing in the run dir, include it with status=missing and a reason.
   - PROMINENTLY surface incomplete runs (e.g., top-level “incomplete_runs” list).

C) Do not change cap semantics or window coverage semantics here — only surface and enforce them.

Step 3 — Tests (must be deterministic and small)
Add unit tests that would have caught the current bug:
1) Partial run detection regression test:
   - Build a temp directory shaped like a partial RC dir (only a resolved_config file or manifest but missing metrics/diagnostics).
   - Assert: summarizer marks is_complete=false, excluded_from_aggregate=true, and reports missing_files non-empty.
2) Missing sections test:
   - Build a minimal fake “rc-lite-sanity” directory with only daily DoW outputs.
   - Assert summary_sanity.json still contains placeholders for vol-state + weekly sections with status=missing (not silently absent).
3) Aggregate exclusion test:
   - Provide one complete + one incomplete run; assert aggregates reflect only the complete run.

Minimum test suite:
- make test-fast
- pytest -m unit -k "summary or summarize_rc_sanity or run_meta"

Record exact commands + pass/fail in docs/agent_runs/<RUN_NAME>/TESTS.md and in commit bodies.

Step 4 — Deterministic integration validation (real-data)
Run:
- EXEC_MODE=deterministic make rc-lite-sanity
Then:
- Run the summarizer on the produced output dir (or whatever rc-lite-sanity uses)
- Verify:
  - summary_sanity.json exists
  - sections exist
  - incomplete runs are listed and excluded
  - cap_active defaults to false for the integration run; if any cap_active=true appears, treat as a failure unless explicitly part of the Make target (it shouldn’t be)

Record the output directory paths in docs/agent_runs/<RUN_NAME>/RESULTS.md.

Step 5 — Documentation updates
- Update PROGRESS.md:
  - include date/time, branch/sha, RUN_NAME, and artifact paths.
  - ensure the sha recorded matches git HEAD at the time of commit (no ambiguity).
- Update project_state/KNOWN_ISSUES.md:
  - mark the “Partial RC dir” issue fixed ONLY if the integration run + tests demonstrate it.

Step 6 — Commits (small + auditable)
- Keep commits small and logical (e.g., completeness core; summarize wiring; tests; docs).
- Every commit MUST include in the commit body:
  - Tests: <exact commands>
  - Artifacts: <paths>

Finish
- Generate the review bundle:
  make gpt-bundle TICKET=ticket-05 RUN_NAME=<RUN_NAME>
- Record the bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md.
