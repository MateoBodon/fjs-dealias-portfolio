You are Codex operating under AGENTS.md (binding). Complete Ticket #5 end-to-end: produce one advisor-ready uncapped daily DoW result table with full validity checks + documentation. Do NOT write a long upfront plan; instead, inspect, execute, validate, and document.

Ticket: ticket-05
Branch: feat/ticket-05-advisor-ready-rc
RUN_NAME format (required): YYYYMMDD_HHMMSS_ticket-05_advisor-ready-rc
Create run log dir at: docs/agent_runs/<RUN_NAME>/

Hard constraints (stop-the-line):
- Do not claim results without pointing to artifacts under reports/ + including exact paths in RESULTS.md.
- Do not generate “headline” summaries if cap_active=true anywhere in the run OR if comparison_valid_* missing/0 OR if n_effective_* < min_comparison_windows without explicit documentation.
- No silent fallbacks: if MV solver is missing, default must fail-loud unless explicitly configured as smoke-only.
- Prefer pinned configs. Do not add ad-hoc CLI knobs unless you also pin them into a committed config file.

Deliverables for this ticket:
1) A headline-eligible daily DoW run (uncapped) with:
   - summary/summary_perf.csv
   - summary/summary_detection.csv
   - summary/overlay_forensics.csv
   - summary/limitations.md
   - summary/completeness.json
   And validation: cap_active=false, comparison_valid_* = 1, n_effective_* >= 50.
2) Update docs:
   - PROGRESS.md: new entry with exact commands + output dir + key deltas and detection rate
   - project_state/CURRENT_RESULTS.md: add the new valid run with the headline deltas (and clearly state config: shrinker, prewhiten, edge_mode, group_design, window/horizon/assets_top)
3) Finish by generating a bundle and recording its path:
   - make gpt-bundle TICKET=ticket-05 RUN_NAME=<RUN_NAME>
   - Record the bundle filepath in docs/agent_runs/<RUN_NAME>/RESULTS.md

Process steps (do in order; record EVERY command in COMMANDS.md):
A) Setup + logging
- git checkout main && git pull && git status -sb (must be clean)
- git checkout -b feat/ticket-05-advisor-ready-rc
- Create docs/agent_runs/<RUN_NAME>/ with required files:
  PROMPT.md (paste this prompt verbatim)
  COMMANDS.md (append as you run commands)
  RESULTS.md (fill as you validate)
  TESTS.md (fill)
  META.md (git sha, config hash, dataset/factor ids+hashes, config hash, env notes)

B) Tests first
- Run: make test-fast
- Record outcome in TESTS.md
- Any failures: fix minimally, small commits, each commit body includes: "Tests run: ..."

C) Mandatory deterministic smoke BEFORE headline evidence (AGENTS baseline)
Because tickets 1–4 changed daily pipeline code, you must run a deterministic daily smoke now and verify summary artifacts exist:
- Run: EXEC_MODE=deterministic make rc-lite-sanity
- Identify the produced reports/<rc_dir> path (from stdout or Makefile conventions)
- Run: PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/<rc_dir>
- In RESULTS.md, paste:
  - the rc_dir path
  - a snippet confirming overlay_forensics.csv exists and is non-empty (show row count)
  - any limitations.md warnings (paste them)

If rc-lite-sanity cannot complete, DO NOT proceed to headline runs. Record the failure clearly in RESULTS.md and stop (this would mean we must do Ticket #6 next).

D) Headline-eligible advisor run (daily DoW, uncapped)
Goal: one clean run with no caps/truncation/max-windows.
Preferred approach:
- Use an existing Make target if it is truly uncapped: make rc-dow
Otherwise run explicitly with a pinned config:
- Use experiments/eval/config.paper_v1.yaml OR create a new pinned config for this run (committed) if needed.
- Run with:
  - group_design=dow
  - window=126, horizon=21, assets_top=60 (paper v1 defaults)
  - shrinker: pick ONE strong baseline (rie preferred) for the headline run unless rc-dow already includes a standard set
  - ensure no --max-windows and no start/end date truncation
- Output to a new timestamped directory: reports/rc-ticket-05-<timestamp>/

Then:
- Run: PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/<run_dir>

E) Validate acceptance criteria (must be explicit in RESULTS.md)
From reports/<run_dir>/run.json and summary/*.csv:
- cap_active == false and cap_sources == [] (or empty)
- summary_perf.csv:
  - comparison_valid_mse == 1, comparison_valid_qlike == 1 (and dm if present)
  - n_effective_mse >= 50 and n_effective_qlike >= 50
- overlay_forensics.csv exists and has >0 rows
- limitations.md has no “excluded headline runs” section for caps or mv-skip
If anything fails:
- Diagnose without “tuning on real data”.
- If the failure is structural (e.g., caps from Makefile defaults), fix config/Makefile so the run is truly uncapped and rerun.
- If the failure is data/coverage driven, document and stop-the-line (no fake headline).

F) Docs updates + commits
- Update PROGRESS.md and project_state/CURRENT_RESULTS.md with:
  - exact run dir
  - config (including prewhiten status)
  - headline deltas (ΔMSE, ΔQLIKE) and detection rate
- Update docs/CODEX_SPRINT_TICKETS.md:
  - mark Ticket #5 as DONE if (and only if) acceptance criteria pass
  - note that Ticket #4 is unblocked if the deterministic smoke in step C passed

Commit rules:
- Small logical commits.
- Every commit body MUST include: "Tests run: <commands>"

G) Bundle for review
- Run: make gpt-bundle TICKET=ticket-05 RUN_NAME=<RUN_NAME>
- Add the resulting bundle path to docs/agent_runs/<RUN_NAME>/RESULTS.md
- End with git status -sb clean.

all current liles that arent code should not be commited but kept on local always, so whatever way is best for that, ignore on .git? then you can push everything to origin as well, then continue with all stept
