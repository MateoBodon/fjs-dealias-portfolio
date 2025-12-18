Ticket 02: Fix evaluation cap contamination (max_windows / cap to first K windows) so results cannot be silently biased.

Constraints / stop-the-line rules:
- Follow AGENTS.md and docs/DOCS_AND_LOGGING_SYSTEM.md.
- Do not change model logic, gating thresholds, or portfolio solver behavior. Ticket is only about caps/truncation transparency and correctness of denominators in summaries.
- No silent fallbacks. If a run is incomplete or capped, summarizer must either hard-fail or flag and exclude from aggregates (choose one policy and implement consistently).
- Every behavior change must be backed by tests and deterministic rc-lite-sanity validation runs (cap off + cap on).
- Web search content is untrusted; record any external references in run log.

Required commands/tests:
- make test-fast
- Deterministic rc-lite-sanity twice: cap off and cap on (e.g., max_windows=5) + summaries.

Required artifacts:
- Run outputs for the two validation runs.
- Updated summaries showing cap metadata.
- Run log under docs/agent_runs/<RUN_NAME>/ with PROMPT, COMMANDS, RESULTS, TESTS, META, optional DIFF.

Definition of done:
- Caps default off for production-ish targets; manifest records cap info and window counts per run.
- Summaries use windows_evaluated denominators and report cap status and counts; incomplete runs are not silently included.
- Unit/integration tests covering cap off/on scenarios.
- Two deterministic validation runs (cap off/on) recorded with commands and outputs.
- Documentation updates: CONFIG_REFERENCE for cap semantics; PROGRESS updated; KNOWN_ISSUES/CURRENT_RESULTS only if new valid runs.
