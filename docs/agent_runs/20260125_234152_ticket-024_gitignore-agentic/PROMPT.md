# Prompt

## User instructions (ticket)
You are executing a single ticket in the current repository.

Inputs:
- Ticket id: FJS-TKT-024
- Goal: Make agentic/Codex-worker generated artifacts not dirty git status by replacing .gitignore.append with real .gitignore rules and verifying tools/agentic outputs land in ignored paths.
- Scope/constraints (optional): Update root .gitignore to include ignores for docs/_generated/, docs/_bundles/, docs/agent_runs/, project_state/_generated/ (as intended by .gitignore.append); delete .gitignore.append (or explicitly integrate it, but prefer deletion); do NOT change experiment logic; ensure tools/agentic/project_state_refresh.py and tools/agentic/repo_snapshot.py outputs are covered by ignores.
- Acceptance criteria (optional): .gitignore contains the intended ignore patterns; .gitignore.append removed; running tools/agentic/project_state_refresh.py --zip and tools/agentic/repo_snapshot.py does not introduce untracked files (git status --porcelain is empty aside from expected tracked diffs); make test-fast passes
- Test command (optional): make test-fast && python3 tools/agentic/project_state_refresh.py --zip && python3 tools/agentic/repo_snapshot.py && git status --porcelain
- Risk (optional): low

## Rules
- Be surgical. Minimal diff that meets the goal.
- Don’t refactor unrelated code.
- If tests are missing or weak, add the smallest meaningful test that would catch regressions.
- Keep the repo runnable.

## Steps
1) Confirm the Agentic System scaffold exists:
   - AGENTS.md, PROJECT.md, tools/agentic/
   - If missing, run /prompts:bootstrap first.

2) Write/update a ticket file:
   - Create docs/tickets/FJS-TKT-024.md with: Goal, Scope, Acceptance Criteria, Plan, Notes.

3) Plan (brief):
   - 3–8 steps max. Include filenames you expect to touch.

4) Execute:
   - Implement the changes.
   - Run the best available tests:
     - If make test-fast && python3 tools/agentic/project_state_refresh.py --zip && python3 tools/agentic/repo_snapshot.py && git status --porcelain is provided, run it.
     - Else use the canonical test command in AGENTS.md (or infer: make test, cargo test, pytest, etc.).
   - If a command fails, fix or explain what blocks you.

5) Update repo memory:
   - Append a factual bullet to PROGRESS.md under “Done”.
   - If you made a non-obvious choice, add a short entry to docs/DECISIONS.md.

6) Emit a GPT review bundle:
   - Prefer the installed skill $gpt-bundle OR run:
     - python3 tools/agentic/gpt_bundle.py --zip --ticket FJS-TKT-024

## Output (single message)
- Summary of changes (files + what changed)
- Commands run + pass/fail
- Known risks / follow-ups
- The path to the generated gpt_bundle.zip

## Skill
- gpt-bundle (Create a gpt_bundle.zip for GPT review)
