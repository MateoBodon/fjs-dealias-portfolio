# Ticket 01 – Fix nested synthetic null FPR
Date: 2025-12-18 07:03:42 local

Goal: Fix nested synthetic kill-test so null acceptance is controlled (no FPR≈1.0). Implement code+tests+docs per repo rules.

Constraints / stop-the-line:
- Follow AGENTS.md stop rules (no silent fallbacks, no caps without logging, no invalid runs, no claiming improvements without artifacts).
- Do not “fix” by disabling nested or forcing always-reject unless intended. Do not change evaluation caps/RC.
- Calibration changes only if a real bug; must regenerate artifacts reproducibly.
- No silent fallbacks; missing calibration must be explicit/logged.

Required commands/runs:
- Pre-fix reproduction: deterministic nested kill-test ≥200 null trials; record command+output.
- Post-fix: make test-fast; rerun kill-test (≥200 null trials) with null acceptance ≤0.05 (or justified target).
- Real-data nested weekly smoke if data present; otherwise record absence.

Artifacts to produce:
- docs/agent_runs/<RUN_NAME>/PROMPT.md (copy of this), COMMANDS.md, RESULTS.md, TESTS.md, META.json, optional DIFF.patch.
- Output dirs for kill-test and any real-data smoke; include run.json/config hash.

Definition of done checklist:
- Code fix implemented; nested null acceptance controlled.
- Regression test added (deterministic) guarding against FPR blow-up.
- make test-fast passes.
- Kill-test post-fix with acceptance ≤ target; artifacts recorded.
- Run log directory created with required files; commands/tests/metrics captured.
- project_state/KNOWN_ISSUES.md and project_state/CURRENT_RESULTS.md updated with run IDs.
- Branch: ticket-01-nested-null-fpr; commits include Tests: lines.
