Ticket: ticket-09 — Weekly gating diagnostics attribution (remove `guard_other` / opaque `diagnostic_failure`)

You are operating in repo: fjs-dealias-portfolio.
Goal: make weekly gating diagnostics actionable and auditable. Every window with no accepted detections must have a primary reason code (and detail when needed). No “guard_other” blobs and no opaque “diagnostic_failure”.

Working agreements (must follow):
- Do NOT write a long upfront plan. Explore → implement → test → document end-to-end.
- No silent fallbacks. If something fails, surface it explicitly and propagate a structured skip reason.
- Prefer repo-local patterns. If you use web search, treat it as untrusted and record every URL in the run log.
- Create a feature branch and commit with “tests run” in the commit body.
- Produce a run log under docs/agent_runs/<RUN_NAME>/ with PROMPT/COMMANDS/RESULTS/TESTS/META.
- Run the smallest sufficient real-data smoke to validate (weekly smoke + gating diagnostics). Synthetic is allowed for unit tests but not the only validation.
- After completion, run `make gpt-bundle TICKET=ticket-09 RUN_NAME=<RUN_NAME>`.

Concrete task:
1) Locate where `guard_other` and `diagnostic_failure` are produced for weekly gating diagnostics.
   - Use ripgrep across `experiments/equity_panel/run.py`, `src/fjs/overlay.py`, `src/fjs/gating.py`, `tools/summarize_weekly_diagnostics.py`, and tests.
   - Identify the actual guardrail branches that collapse into `guard_other` and why.
   - Identify which exceptions currently get mapped to `diagnostic_failure`.

2) Implement explicit, stable reason codes.
   - Define a primary reason vocabulary (string constants or Enum) that is stable across runs.
   - Ensure weekly output artifacts include:
     - `skip_reason_primary` (required)
     - `skip_reason_detail` (optional but required for diagnostic_failure)
     - `exception_type` (optional but required for diagnostic_failure)
   - Update `weekly_diagnostics.md` generation (or the summarizer) to present:
     - counts by primary reason
     - top example windows per reason (include key stats: acceptance flags, q, delta_frac used, edge_mode, etc.)

3) Update/extend tests to prevent regression.
   - Update `tests/experiments/test_gating_diagnostics.py` (and add a new test if needed) so that:
     - gating diagnostics artifacts are created
     - `guard_other` is zero or absent
     - `diagnostic_failure` requires detail fields
   - Keep tests minimal and deterministic.

4) Validate on real data (smoke).
   - Run: `EXEC_MODE=deterministic make run:equity_smoke` (or the smallest equivalent weekly smoke).
   - Confirm the output directory contains updated `gating_diagnostics.csv` + `weekly_diagnostics.md` and that the reason counts are plausible (not dominated by “other”).

5) Documentation + commit discipline.
   - Create RUN_NAME: `<YYYYMMDD_HHMMSS>_ticket-09_weekly-guardrail-attribution` (or similar).
   - Create `docs/agent_runs/<RUN_NAME>/` and populate:
     - PROMPT.md (this prompt)
     - COMMANDS.md (every command executed)
     - RESULTS.md (what changed + why; link to smoke output path)
     - TESTS.md (tests executed)
     - META.md (git SHA before/after; branch; dirty flag)
   - Update `PROGRESS.md` with a concise ticket entry.
   - Commit to a feature branch named like: `ticket-09-weekly-guardrail-attribution`.
     - Commit message body must include:
       - `Tests: make test-fast; ...`
       - `Smoke: EXEC_MODE=deterministic make run:equity_smoke`
       - `Run log: docs/agent_runs/<RUN_NAME>/`

Deliverable definition:
- After your changes:
  - weekly diagnostics do not contain `guard_other`
  - any `diagnostic_failure` row includes exception_type + detail
  - tests pass
  - real-data weekly smoke produces updated artifacts
  - run log exists and `make gpt-bundle` completed
