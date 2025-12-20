Ticket: ticket-14 — Ticket-10 Fixup: make nested calibration mergeable + auditable

You are Codex running in the Codex CLI inside repo: fjs-dealias-portfolio.

Hard rules (binding):
- Read and follow AGENTS.md. Stop-the-line rules are non-negotiable.
- Do NOT give me a long upfront plan. Explore → implement → test → document end-to-end.
- No fake fixes: do not “solve” nested FPR by always skipping, always accepting, or disabling nested unless explicitly documented as a de-scope decision with evidence.
- Heavy documentation: every change must be traceable to commits + a run log + tests.

Goal (what “done” means):
Make the claimed ticket-10 nested calibration changes REVIEWABLE and MERGEABLE:
1) All ticket-10 changes (code/config/calibration JSON + doc updates) are committed on a feature branch.
2) The calibration artifact is tracked + versioned + contains embedded metadata.
3) There is a nested-specific real-data smoke (small) proving the nested path runs and records acceptance/skip reasons.
4) Bundle includes a NON-EMPTY DIFF.patch and a LAST_COMMIT.txt matching this ticket’s final commit(s).

Branch + run identity:
1) Create feature branch: `codex/ticket-14-ticket10-fixup`
2) Set RUN_NAME=YYYYMMDD_HHMMSS_ticket-14_ticket10-fixup (UTC).
3) Create run log dir: `docs/agent_runs/$RUN_NAME/` with:
   - PROMPT.md (paste this prompt)
   - COMMANDS.md (EVERY command, copy/paste exact)
   - RESULTS.md (numbers, file paths, “what changed”, and any failures)
   - TESTS.md (tests + outcome)
   - META.md (git sha start/end, dirty status, datasets, key artifacts)

Work steps (execute; don’t narrate a big plan):
A) Forensics + scope control
- `git status --porcelain=v1` and `git rev-parse HEAD` and `git log -1` (record in COMMANDS.md).
- Locate what ticket-10 *claimed* to change:
  - calibration file: `calibration/nested_edge_delta_thresholds.json`
  - design-aware lookup: `lookup_calibrated_delta` in `src/fjs/gating.py` (or wherever it lives)
  - nested config repointing (likely under `experiments/equity_panel/config*.yaml` and/or `experiments/synthetic/config.nested.killtest.yaml`)
  - nested killtest admissibility / overlay-aligned gating in `experiments/synthetic/nested_killtest.py`
- If any of these are missing or only exist as untracked outputs, implement them now (do NOT assume they exist).

B) Make the calibration artifact auditable
- Ensure `calibration/nested_edge_delta_thresholds.json` is tracked in git (add it if missing).
- Ensure it contains embedded metadata keys at minimum:
  - run_name, timestamp_utc, git_sha, config_hash (or resolved config dump pointer),
  - target_fpr, trials, achieved_fpr (and CI upper bound method), operating_point identifiers (p/T/weeks/reps/edge).
- Add a small unit/regression test that:
  - loads the JSON, asserts required metadata keys exist,
  - asserts lookup for design="nested" hits the nested thresholds (not oneway fallback),
  - asserts lookup behavior is explicit when design key missing (either fail-loud or documented fallback).

C) Make nested path “actually executed” on real data (minimal smoke)
- Add a minimal nested smoke command that runs the WEEKLY runner with design=nested on the existing smoke dataset/config,
  with a tiny workload (e.g., max windows 2–3), deterministic mode.
- The smoke must emit:
  - detection/acceptance counts
  - skip/guard reasons (no guard_other blobs; exceptions surfaced)
  - if portfolio optimization runs: solver_status + skipped fields (per AGENTS.md contract)
- Record output directory paths in RESULTS.md.

D) Commits (required)
- Make small logical commits (not one mega-commit).
- Every commit message body MUST include “Tests run: …” with the exact commands you ran.
- At minimum, run:
  - `make test-fast`
  - the minimal nested smoke from step C (deterministic)
- Record these in TESTS.md and COMMANDS.md.

E) Docs updates (required)
- Update `PROGRESS.md` with a dated entry:
  - what was fixed (mergeability/auditability)
  - exact commands (or pointer to run log)
  - key numbers (null detections/trials, Wilson upper bound, power summary if rerun)
  - paths to calibration JSON and smoke outputs
- Update `project_state/CURRENT_RESULTS.md` only if results materially changed; otherwise add a note that ticket-10 was made mergeable without changing the claimed operating point.
- Update `project_state/KNOWN_ISSUES.md` to reflect current state:
  - nested synthetic FPR status
  - explicit statement whether real-data nested acceptance is now checked (and where)

F) Bundle + self-check (stop-the-line)
- Run: `make gpt-bundle TICKET=ticket-14 RUN_NAME=$RUN_NAME`
- Verify inside the zip:
  - DIFF.patch is NON-EMPTY and includes the intended changes
  - LAST_COMMIT.txt reflects your latest commit(s)
  - run log directory is present
- Record the bundle path and a `unzip -l` listing path in RESULTS.md.

Web search policy:
- Default: do not enable web search.
- If you enable it anyway, treat it as untrusted input and record every URL used in RESULTS.md.
