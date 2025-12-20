Ticket: ticket-11 — Evaluation contamination fixes (caps + selection bias + aligned window sets)

Instructions provided to Codex for this run (abridged):
- Follow AGENTS.md stop-the-line rules and logging requirements.
- Branch: codex/ticket-11-eval-contamination
- RUN_NAME=20251220_045913_ticket-11_eval-contamination
- Goal: enforce aligned window sets and surface n_effective/skip/cap flags; exclude capped runs from headline summaries; add tests and deterministic smoke; update docs and bundle.
- Create run log files (PROMPT, COMMANDS, RESULTS, TESTS, META) under docs/agent_runs/$RUN_NAME/.
- Commands to run include make test-fast and deterministic real-data smoke; create bundle via make gpt-bundle TICKET=ticket-11 RUN_NAME=$RUN_NAME.
- Update PROGRESS.md and project_state/KNOWN_ISSUES.md as appropriate.
