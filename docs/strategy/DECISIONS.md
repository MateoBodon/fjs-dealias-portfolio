# Decisions

last_updated: 2026-07-03
updated_by: Codex T-000
source_event: T-000 install AI Project OS v2

## 2026-07-03 - Install AI OS v2 Without Moving Legacy Docs

- Decision: Copy and index pre-v2 docs under `docs/_archive/pre_ai_os_v2/20260703/` instead of moving or deleting them.
- Rationale: Existing paths are referenced by run logs, bundle tooling, historical reviews, and human workflows.
- Consequence: Old docs stay accessible at original paths and in the archive snapshot, but v2 docs under `docs/strategy/` are the current strategy surface.

## 2026-07-03 - Separate Pro State Bundles From Heavy Review Bundles

- Decision: Add `tools/agentic/ai_os_bundle.py` with separate `project_state_audit` and `review` profiles.
- Rationale: Pro needs project state and strategic context; Heavy needs ticket delta, commands, validation, and residual risk.
- Consequence: Generated bundles live under `reports/_bundles/` and have their own manifests.

## 2026-07-03 - Do Not Rewrite Product Behavior

- Decision: T-000 changes are limited to docs, archive/state artifacts, run logs, and bundle tooling.
- Rationale: The ticket is infrastructure/state setup, not a research or product behavior ticket.
- Consequence: Research code, algorithms, data, and existing validation semantics remain unchanged.

## Pre-v2 Decisions Preserved

The pre-v2 decision log remains in `docs/DECISIONS.md` and is copied into the archive snapshot. It includes prior decisions about repo-bootstrap, gpt-bundle wrappers, dirty-tree bundle metadata, and run-log schema alignment.
