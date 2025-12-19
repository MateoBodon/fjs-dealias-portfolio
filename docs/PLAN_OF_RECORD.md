# Plan of Record

Purpose: keep a concise snapshot of the currently accepted scope for the FJS-style de-aliasing pipeline and evaluation harness.

Source of truth:
- docs/LONG_TERM_PLAN.md (experiment grid and long-horizon objectives)
- PROJECT_STATE/*.md (status, configs, known issues)
- PROGRESS.md (checkpointed updates)

As of 2025-12-19 (RUN_NAME: 20251219_072353_ticket-06_gpt-bundle-restore):
- Maintain FJS-inspired detection/overlay on covariance estimators (src/fjs, src/finance) with config-driven experiment runners.
- Preserve synthetic calibration harness and artifacts under calibration/ and reports/synthetic/.
- Support equity-panel weekly designs (DoW, nested, vol-state) with configs under experiments/equity_panel/.
- Keep evaluation harness (experiments/eval/, src/evaluation/) aligned with RC outputs and gating defaults.

Process:
- When scope or experiment grids change, update docs/LONG_TERM_PLAN.md + PROJECT_STATE/* and mirror the decision here.
- Record changes in PROGRESS.md alongside run/bundle identifiers.
