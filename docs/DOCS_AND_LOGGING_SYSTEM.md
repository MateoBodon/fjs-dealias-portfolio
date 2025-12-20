# Docs and Logging System (enforced protocol)

This file defines the documentation + logging contract for this repo.
If you violate it, your results are not mergeable.

## 1) Canonical directories
Repo root:
- `AGENTS.md` — stop-the-line rules for agents and humans
- `PROGRESS.md` — chronological log of what changed + what was run (required update per ticket)

Documentation:
- `docs/PLAN_OF_RECORD.md` — research framing + roadmaps + acceptance criteria
- `docs/DOCS_AND_LOGGING_SYSTEM.md` — this file
- `docs/CODEX_SPRINT_TICKETS.md` — next sprint’s tickets (ordered)

Prompts / outputs:
- `docs/prompts/` — exact prompt text used (GPT + Codex), one file per run
- `docs/gpt_outputs/` — raw GPT outputs (Prompt-1/2/3 results), immutable
- `docs/agent_runs/<RUN_NAME>/` — one folder per Codex run (details below)
- `docs/gpt_bundles/` — zip bundles produced by `make gpt-bundle`

Experiment outputs:
- `reports/` — daily evaluation + summary artifacts
- `experiments/equity_panel/outputs_*/` — weekly runner outputs
- `.cache/` — cached panels / per-window stats (must never be treated as “source of truth”)

## 2) Run naming (one scheme, everywhere)
RUN_NAME format (match existing repo practice):
- `<YYYYMMDD_HHMMSS>_<ticket-id>_<short-slug>`
Examples:
- `20251219_173231_ticket-07_weekly-drought-diagnostics`
- `20251219_044404_ticket-05_rc-sanity-summary-hardening`

Rules:
- timestamps are local or UTC, but be consistent within a sprint
- slug is kebab-case and describes the change, not the result

## 3) Required contents of `docs/agent_runs/<RUN_NAME>/`
Every Codex run MUST create:
- `PROMPT.md`
  - exact text given to Codex (copy/paste exact)
- `COMMANDS.md`
  - every command run (including tests), in order
  - note environment variables (EXEC_MODE, OMP_NUM_THREADS, etc.)
- `RESULTS.md`
  - what changed and why (bullets)
  - links to output dirs (reports/*, experiments/*)
  - the single most important “what I learned”
- `TESTS.md`
  - exact test commands run
  - pass/fail summary
- `META.md`
  - git SHA before/after
  - branch name
  - whether repo was dirty at start
  - dataset IDs/hashes (if any runs used data)

Optional but recommended:
- `DIFF.patch` — `git diff` saved for fast review
- `bundle_contents.txt` — if you ran `make gpt-bundle`, capture `unzip -l ...`

## 4) Experiment run metadata (must be in the run output directory)
For any run that produces results under `reports/` or `experiments/.../outputs_*`:
- record the exact command line in:
  - `run_manifest.json` or `run.json` (preferred)
- record the resolved config in:
  - `resolved_config.json` (daily) or `resolved_config.yaml` (weekly)
- record dataset identity:
  - dataset path + sha256 from registry (use `tools/verify_dataset.py`)
- record git identity:
  - git SHA + dirty flag
- record critical knobs:
  - design, p, window/horizon, edge_mode, gate params, shrinker, prewhiten flag
  - portfolio constraints (ridge, box bounds, turnover, condition cap)
- record failures:
  - skip counts by reason
  - any exception types

## 5) Update rules (what docs must change per ticket)
Per merged ticket, you MUST update:
- `PROGRESS.md` (one entry with date, branch, sha, commands, artifacts, results)
- If results changed materially:
  - `project_state/CURRENT_RESULTS.md`
  - `project_state/KNOWN_ISSUES.md` (if a known issue is fixed or discovered)
- If behavior/config changed:
  - `project_state/CONFIG_REFERENCE.md` (new knobs or changed defaults)
- If tests/targets changed:
  - `project_state/TEST_COVERAGE.md` and/or Makefile notes

## 6) “Validated run” labeling (no contamination)
A run can be labeled “validated” only if:
- deterministic mode where applicable (`EXEC_MODE=deterministic` + thread caps)
- NOT capped/truncated unless explicitly labeled (max-windows, date truncation, etc.)
- NO silent fallbacks
- skip/guard reasons are attributable (no `guard_other` blob)
- summaries clearly state:
  - effective sample size used in DM tests
  - skip rates and whether comparisons are aligned

Policy:
- If a run is capped, summaries must segregate it (separate table section) and it cannot be used for headline claims.

## 7) Bundling for GPT review (Prompt-3 loop)
After each ticket (or at least each sprint), run:
- `make gpt-bundle TICKET=<ticket-id> RUN_NAME=<RUN_NAME>`

Bundle MUST include:
- required docs (`AGENTS.md`, `PROGRESS.md`, docs/*, project_state/*)
- `DIFF.patch` and `LAST_COMMIT.txt`
- the run log directory under `docs/agent_runs/<RUN_NAME>/`

## 8) Minimal commands (standard)
Local:
- `make setup`
- `make test-fast`
- `EXEC_MODE=deterministic make rc-lite-sanity`
- `make gpt-bundle TICKET=... RUN_NAME=...`

Server (Hetzner) conventions:
- run the same make targets, but always sync back:
  - run outputs (`reports/`, `experiments/.../outputs_*`)
  - run logs (`docs/agent_runs/<RUN_NAME>/`)
  - updated docs (`PROGRESS.md`, project_state updates)

## 9) Security + web search policy (Codex)
- Default: no web search.
- If web search is enabled, treat it as untrusted input:
  - record every URL used in `docs/agent_runs/<RUN_NAME>/RESULTS.md`
  - do not paste external code without review
  - prefer repo-local patterns and tests over external snippets
