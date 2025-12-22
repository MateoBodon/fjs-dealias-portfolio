You are Codex operating under AGENTS.md (binding). Complete Ticket #6 end-to-end: fix the daily eval runner so an uncapped daily DoW “paper v1” run is not incorrectly flagged as cap_active due to window_coverage / holdout_empty windows, and produces non-empty headline summary tables.

Do NOT write a long upfront plan. Instead: inspect → reproduce → implement → test → validate on real-data smoke → document → bundle.

Ticket: ticket-06
Branch: feat/ticket-06-window-coverage
RUN_NAME format (required): YYYYMMDD_HHMMSS_ticket-06_window-coverage
Create run log dir at: docs/agent_runs/<RUN_NAME>/

Hard constraints (stop-the-line; enforce strictly):
- Do NOT “fix” by disabling caps, forcing cap_active=false, or re-including capped runs in headline tables.
- No silent window dropping: if windows are excluded from evaluation planning, you must log counts + reasons in run.json and surface them in summary/limitations.md.
- No data tampering: do not hand-edit data/*.csv. If you must create a derived dataset, do it via a script + registry update + hashes.
- No merge without tests + logs: at minimum run and record `make test-fast`. Every commit body must include “Tests run: …”.

Deliverables:
1) Code fix + tests:
   - An uncapped daily eval run using experiments/eval/config.paper_v1.yaml with daily DoW no longer ends up with cap_active=true due solely to holdout_empty/window planning.
   - Add a CI-safe regression test that would have failed before this fix.
2) Real-data validation:
   - Run a deterministic daily DoW paper-v1 eval on real data (no max-windows; no start/end truncation; assets_top/window/horizon per config) and confirm:
     - cap_active == false
     - cap_sources empty (or absent)
     - summary/summary_perf.csv and summary/summary_detection.csv are NON-EMPTY
     - comparison_valid_* == 1 and n_effective_* >= 50 (or explicitly document why not)
3) Documentation:
   - Update PROGRESS.md with commands, outputs, and the cap/window_coverage fix rationale.
   - Update project_state/KNOWN_ISSUES.md (remove/resolve if fixed) and/or CURRENT_RESULTS.md if the run becomes headline-eligible.
   - Update docs/CODEX_SPRINT_TICKETS.md:
     - Mark Ticket #5 as FAIL (window_coverage cap blocked headline run).
     - Add/expand Ticket #6 as DONE only if this ticket truly passes.

Required run log (docs/agent_runs/<RUN_NAME>/):
- PROMPT.md (this prompt verbatim)
- COMMANDS.md (every command executed, copy/pasteable)
- RESULTS.md (explicit checks + exact artifact paths)
- TESTS.md (test outputs summarized)
- META.md (git sha, config hash, dataset hashes, env notes)

Process (do in order; log every step):
A) Setup + logging
- git checkout main && git pull && git status -sb (must be clean)
- git checkout -b feat/ticket-06-window-coverage
- Create docs/agent_runs/<RUN_NAME>/{PROMPT,COMMANDS,RESULTS,TESTS,META}.md and paste this prompt into PROMPT.md.

B) Tests first
- Run: make test-fast
- Record results in TESTS.md.
- If failures: fix minimally with small commits.

C) Reproduce the failure mechanism (use prior evidence, but confirm on current HEAD)
The ticket-05 bundle reported:
- cap_active=true due to cap_sources=['window_coverage'] on an “uncapped” paper-v1 DoW run
- windows_evaluated < windows_requested and many missing windows had reason_code=holdout_empty
Your job is to reproduce on a fresh run OR reuse an existing reports/ directory if present, but you must show evidence in RESULTS.md:
- A snippet from run.json showing cap_active and window_coverage fields pre-fix.
- Counts of reason_code==holdout_empty (or equivalent) and whether those rows have NaN window identifiers.

D) Implement the fix (research-valid)
Goal: windows_requested should mean “windows that are evaluable given data + horizon”, not “candidate endpoints including impossible holdouts”.
Implement a transparent window planning / accounting change so that:
- Windows that cannot be evaluated because the holdout horizon is empty are excluded from the requested/planned set (so they don’t trigger window_coverage caps),
- BUT you still log how many were excluded and why.

Concrete requirements:
- In experiments/eval/run.py (and any helper it uses), identify where:
  - candidate windows are generated,
  - holdout slicing happens,
  - windows_requested / windows_evaluated / window_coverage are computed,
  - cap_active / cap_sources are assigned.
- Change the accounting to include fields like:
  - windows_candidate (optional)
  - windows_planned / windows_requested (post-feasibility)
  - windows_dropped_holdout_empty (count)
  - windows_dropped_reasons (dict)
- Ensure diagnostics for dropped windows do not silently disappear: either
  - write them to a separate CSV (preferred), OR
  - include them in diagnostics_detail.csv with non-NaN identifiers,
  but they must NOT force cap_active for headline eligibility when the only issue is “no holdout exists”.

E) Add regression tests (CI-safe)
- Add/extend a test under tests/experiments/ that:
  - builds a tiny synthetic returns dataset with known date index length N,
  - sets window and horizon so that some candidate windows near the end have no holdout,
  - runs the daily eval runner (or its window planner) and asserts:
    - cap_active is false (no explicit caps used),
    - windows_requested == windows_evaluated,
    - windows_dropped_holdout_empty is present and > 0,
    - limitations/completeness (if produced in test) surface the drop count.
- Keep runtime small (few dates, few assets, workers=1, deterministic if applicable).

F) Real-data smoke (must be uncapped)
Run a real-data paper-v1 DoW eval with explicit dataset paths (ticket-05 showed run.py requires them):
- Use:
  - --config experiments/eval/config.paper_v1.yaml
  - --returns-csv data/returns_daily.csv
  - --factors-csv data/factors/ff5mom_daily.csv
  - --out reports/rc-ticket-06-<timestamp>/dow-paper-v1  (or repo convention)
- Then:
  - PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-06-<timestamp>
Validate in RESULTS.md with explicit file paths and snippets:
- run.json: cap_active false, cap_sources empty, window_coverage==1 (or windows_requested==windows_evaluated)
- summary/summary_perf.csv rowcount > 0
- summary/summary_detection.csv rowcount > 0
- summary/overlay_forensics.csv rowcount > 0
- summary/limitations.md has NO “excluded capped runs” entry for this run
- n_effective_* >= 50 and comparison_valid_* == 1 (or explicitly document why not)

If the run is still excluded:
- STOP.
- Do NOT tune thresholds on real data.
- Document the remaining blocker in project_state/KNOWN_ISSUES.md and RESULTS.md.

G) Docs updates + commits
- Update PROGRESS.md with:
  - exact commands
  - exact output dirs
  - before/after explanation of window_coverage/holdout handling
- Update project_state/KNOWN_ISSUES.md and/or CURRENT_RESULTS.md if this changes headline eligibility.
- Update docs/CODEX_SPRINT_TICKETS.md:
  - Ticket #5 = FAIL (window_coverage cap blocked advisor-ready run)
  - Ticket #6 status based on whether acceptance passes

Commit rules:
- Small logical commits.
- Every commit body MUST include: “Tests run: …”
- Keep working tree clean at end.

H) Bundle for review
- Run: make gpt-bundle TICKET=ticket-06 RUN_NAME=<RUN_NAME>
- Record the resulting bundle filepath in docs/agent_runs/<RUN_NAME>/RESULTS.md
- End with: git status -sb (must be clean)
