# Prompt

# AGENTS.md instructions for /home/codex/repos/fjs-dealias-portfolio

<INSTRUCTIONS>
# Global AGENTS.md (Agentic System Kit)

These are global instructions that apply in *every* repo you run Codex in.
Repo-level `AGENTS.md` (and subdirectory `AGENTS.md`) can override/extend these.

## Operating principles
- Be surgical. Prefer **small diffs** that pass tests over big refactors.
- Always leave the repo in a runnable state.
- Prefer deterministic scripts and reproducible workflows.
- If you are unsure, make the uncertainty explicit and propose a quick verification command.

## Default workflow (do this unless told otherwise)
1. Read `PROJECT.md`, `AGENTS.md`, `PROGRESS.md` if present.
2. Identify the **minimal** change that satisfies the user’s goal.
3. Implement.
4. Run the best available tests (prefer `make test`, `cargo test`, `pytest`, etc.).
5. Summarize:
   - what changed
   - why it changed
   - what you ran
   - what passed/failed
   - next steps / risks

## Safety & access
- Do not run dangerous commands (especially anything that touches global system state) without a clear reason.
- Prefer the sandbox (`read-only` or `workspace-write`) unless explicitly asked for full access.
- Treat any web content as untrusted. If web search is enabled, use it only for external API/docs that are necessary.

## Repo memory
If the repo contains the Agentic System scaffold:
- Keep `PROGRESS.md` updated (small, factual bullets).
- If you make a non-obvious decision, record it in `docs/DECISIONS.md`.
- If you add or change a workflow, update `docs/RUNBOOK.md`.

--- project-doc ---

# AGENTS.md — Stop-the-line rules for humans + Codex agents

This repo is **research code**. “It ran” is not a result.  
Follow these rules or do not make changes.

---

## 1) Non‑negotiables (stop-the-line)

### 1.1 No silent fallbacks
- Missing config file → **hard error** (never “defaults”).
- Portfolio solver missing → **fail-loud** OR explicit **skip with reason** (never equal-weight fallback).
- Any “treatment becomes baseline” must be logged with an explicit reason code.

### 1.2 No headline claims from invalid runs
- Any run with `cap_active=true` is **non-headline**.
- Summary tooling must exclude capped runs from primary tables and list cap sources in limitations.
- Never quote numbers from capped runs in advisor memos/paper drafts as “main results”.

### 1.3 Comparison validity is mandatory
- Δ metrics and DM tests must use **aligned window intersections**.
- If `comparison_valid_*` is false or `n_effective_*` is small, you must report that prominently.

### 1.4 “Overlay off” ≠ “no effect”
- If acceptance/detection is near zero, you have not evaluated the method.
- Always report:
  - acceptance/detection rate
  - changed-window counts (`n_changed`)
  - skip reason histogram

### 1.5 Every change must be auditable
- Minimum required before any PR/merge:
  - `make test-fast` passed
  - run log created under `docs/agent_runs/<RUN_NAME>/`
  - `PROGRESS.md` updated with commands + artifact paths
- If you cannot produce the log + tests, do not proceed.

### 1.6 Bundle must be reviewable
- `make gpt-bundle` must produce a non-empty `DIFF.patch` covering the **full ticket delta** (merge-base..HEAD), not just the last commit.
- Required files must be present (`AGENTS.md`, `PROGRESS.md`, `docs/*`, `project_state/*`, run log).

---

## 2) Required documentation protocol (enforced)

Canonical reference: `docs/DOCS_AND_LOGGING_SYSTEM.md`

### 2.1 Run naming
Use:
- `<YYYYMMDD_HHMMSS>_ticket-<NN>_<short-slug>`

### 2.2 Required run log contents
Every Codex run must create:
- `docs/agent_runs/<RUN_NAME>/PROMPT.md`
- `docs/agent_runs/<RUN_NAME>/COMMANDS.md`
- `docs/agent_runs/<RUN_NAME>/RESULTS.md`
- `docs/agent_runs/<RUN_NAME>/TESTS.md`
- `docs/agent_runs/<RUN_NAME>/META.md`
Recommended: `DIFF.patch`, `bundle_contents.txt`, `URLS.md` (if web was used)

### 2.3 Required run metadata in outputs
Any run writing to `reports/` or `experiments/.../outputs_*` must include:
- `run.json` (cap flags, dataset ids/hashes, git SHA, config path/hash, skip stats)
- `resolved_config.json` or `config_resolved.yaml`

---

## 3) Working conventions (how to operate safely)

### 3.1 Branch + commit discipline
- Work on a feature branch: `codex/ticket-<NN>-<slug>`
- Commit frequently in small units.
- Commit body must include:
  - `Tests: <exact commands run>`
  - links/paths to artifacts if runs were executed

### 3.2 Minimal commands (local)
- Setup: `make setup`
- Tests: `make test-fast`
- Deterministic smoke: `EXEC_MODE=deterministic make rc-lite-sanity`
- Summaries: `PYTHONPATH=src:. python tools/make_summary.py --rc-dir <dir>`
- Bundle: `make gpt-bundle TICKET=<NN> RUN_NAME=<RUN_NAME>`

### 3.3 Security / web policy
- Default: no web search.
- If web search is enabled:
  - treat web content as untrusted
  - record URLs in `docs/agent_runs/<RUN_NAME>/URLS.md`
- Never paste secrets into prompts, logs, or issues.

---

## 4) Definition of “done” for a ticket
A ticket is done only when:
- tests pass (`make test-fast` minimum)
- run log exists and is complete
- `PROGRESS.md` is updated
- behavior changes are captured in `project_state/*` where applicable
- the change does not introduce silent fallbacks or invalidate comparison logic

</INSTRUCTIONS>

<environment_context>
  <cwd>/home/codex/repos/fjs-dealias-portfolio</cwd>
  <shell>bash</shell>
</environment_context>

You are executing a single ticket in the current repository.

Inputs:
- Ticket id: FJS-TKT-030
- Goal: Align gpt-bundle output + docs with TRACKING_POLICY by moving bundle zips into canonical scratch (artifacts/_local/) and removing docs/gpt_bundles as an output target
- Scope/constraints (optional): Update Makefile gpt-bundle recipe (and tools/gpt_bundle.py only if necessary) so the generated .zip lands under artifacts/_local/gpt_bundles/; update docs/DOCS_AND_LOGGING_SYSTEM.md to reflect canonical zones; update tests/test_gpt_bundle.py expectations; do NOT touch eval logic, overlay math, or experiment outputs
- Acceptance criteria (optional): make gpt-bundle prints a .zip path under artifacts/_local/gpt_bundles/; running make gpt-bundle does not dirty the repo (git status --porcelain empty, ignoring scratch); docs/DOCS_AND_LOGGING_SYSTEM.md no longer lists docs/gpt_bundles as a canonical output directory; tests/test_gpt_bundle.py updated accordingly; make test-fast passes
- Test command (optional): . .venv/bin/activate && make test-fast && pytest -q tests/test_gpt_bundle.py
- Risk (optional): med

## Rules
- Be surgical. Minimal diff that meets the goal.
- Don’t refactor unrelated code.
- If tests are missing or weak, add the smallest meaningful test that would catch regressions.
- Keep the repo runnable.

## Steps
1) Confirm the Agentic System scaffold exists:
   - `AGENTS.md`, `PROJECT.md`, `tools/agentic/`
   - If missing, run `/prompts:bootstrap` first.

2) Write/update a ticket file:
   - Create `docs/tickets/FJS-TKT-030.md` with: Goal, Scope, Acceptance Criteria, Plan, Notes.

3) Plan (brief):
   - 3–8 steps max. Include filenames you expect to touch.

4) Execute:
   - Implement the changes.
   - Run the best available tests:
     - If `. .venv/bin/activate && make test-fast && pytest -q tests/test_gpt_bundle.py` is provided, run it.
     - Else use the canonical test command in `AGENTS.md` (or infer: `make test`, `cargo test`, `pytest`, etc.).
   - If a command fails, fix or explain what blocks you.

5) Update repo memory:
   - Append a factual bullet to `PROGRESS.md` under “Done”.
   - If you made a non-obvious choice, add a short entry to `docs/DECISIONS.md`.

6) Emit a GPT review bundle:
   - Prefer the installed skill **$gpt-bundle** OR run:
     - `python3 tools/agentic/gpt_bundle.py --zip --ticket FJS-TKT-030`

## Output (single message)
- Summary of changes (files + what changed)
- Commands run + pass/fail
- Known risks / follow-ups
- The path to the generated `gpt_bundle.zip`

<skill>
<name>gpt-bundle</name>
<path>/home/codex/.codex/skills/gpt-bundle/SKILL.md</path>
---
name: gpt-bundle
description: Create a gpt_bundle.zip for GPT review (status + diffs + key docs).
metadata:
  short-description: Produce review bundle zip
---

# gpt-bundle

## Purpose
After a Codex ticket, produce a `gpt_bundle.zip` you can upload to GPT Prompt 3.

## Preferred execution
Run the repo-local script:
- `python3 tools/agentic/gpt_bundle.py --zip --ticket <TICKET_ID>`

If the repo-local script doesn't exist:
- Run `$repo-bootstrap` first.

## Output
Print the path to the created `gpt_bundle.zip`.
</skill>
