good work, for now go through this checklist and make sure it is complete, after, commit, merge to main, push to origin, then after al done make the final complete bundle for the ticket, Core validity

 make run:equity_nested_smoke_tiny output shows zero calibration_missing_p_T skips (check skip histogram / diagnostics).

 Calibration update is not a “fake fix” (no silent mapping). If any approximation exists, it is explicitly logged and backed by synthetic null‑FPR evidence.

 Synthetic results for the newly added (p,T) cells show null‑FPR ≤ target (≤2% or explicitly justified).

Auditability / protocol

 docs/agent_runs/<RUN_NAME>/ contains PROMPT/COMMANDS/RESULTS/TESTS/META (+ DIFF.patch).

 PROGRESS.md updated with exact commands + artifact paths.

 project_state/KNOWN_ISSUES.md updated if the blocker is fixed; CURRENT_RESULTS.md updated if results materially changed.

 Commit bodies include Tests: lines with exact commands.

Repo hygiene

 No large run artifacts accidentally committed (especially under reports/), unless explicitly intended and justified.

 New/updated calibration JSON includes metadata (run_name, timestamp, git SHA, config hash).

Bundle review loop

 make gpt-bundle ... output zip exists and includes a non-empty top-level DIFF.patch. If still empty, prioritize Ticket #21 to fix bundling.
