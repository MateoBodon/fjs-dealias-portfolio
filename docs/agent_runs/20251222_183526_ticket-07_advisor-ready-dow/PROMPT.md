You are Codex operating under AGENTS.md (binding). Complete Ticket #7 end-to-end.

Ticket #7 goal:
Re-run the advisor-ready daily DoW paper-v1 run (uncapped) after the Ticket #6 window-planning fix, produce headline-eligible summary artifacts, update PROGRESS.md + project_state/CURRENT_RESULTS.md, and generate a review bundle.

DO NOT write a long upfront plan. Do: inspect → run → validate → document → bundle.

Branch + run log requirements:
- Create a feature branch: feat/ticket-07-advisor-ready-dow
- RUN_NAME must be: YYYYMMDD_HHMMSS_ticket-07_advisor-ready-dow
- Create run log dir: docs/agent_runs/<RUN_NAME>/
  - PROMPT.md (this prompt verbatim)
  - COMMANDS.md (EVERY command executed, copy/pasteable, no “...” omissions)
  - RESULTS.md (explicit checks + exact artifact paths + key numbers)
  - TESTS.md (tests run + pass/fail + runtimes)
  - META.md (git SHA, config hash, dataset hashes/ids, exec mode, environment notes)

Stop-the-line rules (must enforce):
- Do NOT “fix” by disabling caps, forcing cap_active=false, or excluding bad outcomes from the headline table.
- No silent fallbacks: MV solver must not silently fallback; missing solver must be explicit skip with reason.
- No data tampering: do not hand-edit data/*.csv.
- No merge without tests: run at least make test-fast and record it in commit bodies as “Tests run: …”.

Work steps (do in this order; log every step in COMMANDS.md):

A) Setup
1) git checkout main && git pull && git status -sb
2) git checkout -b feat/ticket-07-advisor-ready-dow
3) RUN_NAME=YYYYMMDD_HHMMSS_ticket-07_advisor-ready-dow
4) mkdir -p docs/agent_runs/$RUN_NAME
5) Create PROMPT/COMMANDS/RESULTS/TESTS/META files. Paste this prompt into PROMPT.md.

B) Tests first
- Run: make test-fast
- Record summary in TESTS.md
- Commit any code/doc changes later; do not commit yet unless you fix something.

C) Run the real-data daily DoW paper-v1 evaluation (UNCAPPED)
- Use the pinned config and real data paths:
  - PYTHONPATH=src:. python experiments/eval/run.py \
      --config experiments/eval/config.paper_v1.yaml \
      --returns-csv data/returns_daily.csv \
      --factors-csv data/factors/ff5mom_daily.csv \
      --out reports/rc-ticket-07-<timestamp>/dow-paper-v1 \
      --exec-mode deterministic
Notes:
- Do NOT set --max-windows, --start, or --end.
- If it takes too long, STOP and document; do not “cap” to make it finish.

D) Build summaries
- PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-07-<timestamp>

E) Validate headline eligibility (must show evidence in RESULTS.md by quoting/snipping exact files)
From reports/rc-ticket-07-<timestamp>/dow-paper-v1/run.json (windows block), confirm:
- cap_active == false
- cap_sources is empty or absent
- window_coverage == 1.0 (or windows_requested == windows_evaluated)
- windows_dropped_holdout_empty is present (>=0) and, if >0, windows_dropped_reasons includes holdout_empty

From reports/rc-ticket-07-<timestamp>/summary/ confirm non-empty:
- summary_perf.csv (rows > 0)
- summary_detection.csv (rows > 0)
- overlay_forensics.csv (rows > 0)
- limitations.md exists and does NOT include a “run capped … excluded” section for this run
Also confirm in summary_perf.csv:
- comparison_valid_* == 1 for the headline rows
- n_effective_* >= 50 (or document explicitly why lower, and then STOP — advisor-ready requires this unless we revise PLAN_OF_RECORD)

F) Create an advisor-readable artifact (small + deterministic)
- Create: reports/rc-ticket-07-<timestamp>/summary/advisor_snapshot.md
Include:
- command used + git SHA
- detection_rate_mean (full regime) and percent_changed
- ΔMSE / ΔQLIKE for EW and MV (full regime)
- one sentence “interpretation” that is limitation-aware (no overclaims)
- link paths to the CSVs in the repo output tree

G) Repo hygiene gates (must run and record outputs in RESULTS.md)
Run the repo’s data/security checks:
1) python3 scripts/check_data_policy.py   (must exit 0)
2) Secret scan:
   - rg -n "WRDS_PASSWORD|WRDS_USERNAME|password|token|secret" -S .
   - If rg is unavailable, use: grep -RInE "WRDS_PASSWORD|WRDS_USERNAME|password|token|secret" .
3) Restricted-data string scan on tracked artifacts:
   - git ls-files | xargs rg -n "strike,.*market_iv|\\bsecid\\b|best_bid|best_ask|best_offer" -S
   - If rg is unavailable, use grep with xargs.
If any hit appears in tracked CSV/parquet-like artifacts: STOP and fix (remove from git, replace with synthetic/public + provenance doc). Do not proceed to bundling until clean.

H) Documentation updates + commits
1) Update PROGRESS.md with:
   - timestamp
   - branch + git SHA
   - exact commands
   - output directories
   - headline metrics + limitations (explicitly note holdout_empty drops if any)
2) Update project_state/CURRENT_RESULTS.md:
   - add/refresh the “Daily DoW paper-v1” entry with the new reports path
   - Fix the YAML front-matter (generated date, git_sha, git_branch, commands) so it matches THIS ticket/run (no stale header).
3) Update docs/CODEX_SPRINT_TICKETS.md:
   - Mark Ticket #7 as DONE with run path + RUN_NAME reference.

Commit rules:
- Use small logical commits (e.g., one for docs updates, one for any code changes if needed).
- Every commit body MUST include: “Tests run: make test-fast”
- Keep working tree clean at the end.

I) Bundle for review
- make gpt-bundle TICKET=ticket-07 RUN_NAME=$RUN_NAME
- Record the produced bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md
- End with: git status -sb (must be clean)
