# DECISIONS

Record non-obvious decisions. Keep it short.

Template:
- Date:
- Decision:
- Context:
- Options considered:
- Why:
- Consequences:

- Date: 2026-01-25
- Decision: Run repo-bootstrap without --force and wrap tools/agentic/gpt_bundle.py to call the repo Makefile target.
- Context: The repo lacked PROJECT.md/tools/agentic, but the bootstrap overlay would overwrite repo-specific AGENTS/PROGRESS and ship a generic gpt_bundle script that conflicts with required merge-base DIFF.patch logic.
- Options considered: Full bootstrap with --force; manual minimal scaffolding.
- Why: Preserve existing repo rules and gpt-bundle auditability while still adding missing scaffold files.
- Consequences: Added scaffold files and a wrapper script; future runs should continue using Makefile gpt-bundle.
