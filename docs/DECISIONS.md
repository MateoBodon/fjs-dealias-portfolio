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

- Date: 2026-01-26
- Decision: Post-process the generated bundle zip to inject `git_dirty` into `BUNDLE_META.md` inside `tools/agentic/gpt_bundle.py`.
- Context: Ticket scope limited changes to the agentic wrapper while requiring `git_dirty` in bundle metadata and no edits to the core diff generator.
- Options considered: Update `tools/gpt_bundle.py`; add a Makefile step; update the zip in the wrapper after `make gpt-bundle`.
- Why: Keeps diff logic untouched and satisfies scope constraints with minimal changes.
- Consequences: The wrapper rewrites the bundle zip to update `BUNDLE_META.md` after creation.
