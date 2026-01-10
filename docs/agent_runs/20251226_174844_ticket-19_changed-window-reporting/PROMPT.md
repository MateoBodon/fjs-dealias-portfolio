You are working in a research repo with strict stop-the-line rules.

Before doing anything: read and follow AGENTS.md and docs/DOCS_AND_LOGGING_SYSTEM.md. If you find a conflict, stop and report it in the run log RESULTS.md.

TICKET: ticket-19
RUN_NAME: use a UTC timestamp + slug exactly like:
  RUN_NAME="$(date -u +%Y%m%d_%H%M%S)_ticket-19_changed-window-reporting"

GOAL (one sentence):
Add conditional (“changed-window only”) effect reporting + weight-change magnitude stats so we can tell whether the overlay matters when it triggers, and ensure this matches the semantics used for n_effective/aligned comparisons.

Hard requirements (do not skip):
- Create a feature branch: codex/ticket-19_changed-window-reporting
- Make small logical commits. Every commit body must include: "Tests: <commands>".
- Run tests: make test-fast (minimum) and record in TESTS.md and commit body.
- Create a complete run log under docs/agent_runs/$RUN_NAME/ with:
  PROMPT.md (paste this prompt verbatim),
  COMMANDS.md (every command you run),
  RESULTS.md (what changed + where outputs are),
  TESTS.md (tests run + pass/fail),
  META.md (git sha start/end, dirty flags, dataset ids/hashes if any real-data run).
- Prefer real-data smoke using the repo’s small derived datasets (fixtures) for speed; synthetic is allowed for unit tests but must not be the only validation.
- Do NOT “fix” by always marking changed=true/false. Changed-window must reflect actual semantics.
- Finish by generating a new bundle and record its path in RESULTS.md:
  make gpt-bundle TICKET=ticket-19 RUN_NAME=$RUN_NAME

Implementation tasks (do end-to-end, no long upfront plan):
1) Inspect how “changed windows” are currently defined and emitted.
   - Find where per-window outputs are written in experiments/eval/run.py (or wherever the eval runner writes metrics_detail / weights / overlay flags).
   - Identify existing fields: accepted/detected flags, any “changed” boolean, n_changed counts, window ids, portfolio ids (EW/MV), and how n_effective_* is computed today.
   - Write down the current semantics in docs/agent_runs/$RUN_NAME/RESULTS.md (short bullets).

2) Define a single, explicit changed-window semantics and make it consistent.
   - Preferred: changed_window := 1 when the treatment run applies a non-noop overlay correction for that window (i.e., “accepted and applied”), else 0.
   - Ensure this is emitted consistently for both EW and MV rows (even if EW weights don’t change; covariance can).
   - If there is already a different semantics in the code, do NOT silently change it. Either (a) keep it and document it clearly, or (b) change it and update tests + PROGRESS.md with the breaking change clearly called out.

3) Add conditional reporting to summaries.
   Update tools/make_summary.py (and tools/summarize_rc_sanity.py if used) so summary tables include:
   - ΔMSE and ΔQLIKE conditional on changed windows only (aligned window intersection restricted to changed==1).
   - n_changed counts and changed fraction (n_changed / n_total_aligned).
   - Weight-change magnitude stats on changed windows:
     - median ||Δw||_2
     - median turnover_delta := sum_i |w_treat - w_base|
   Produce these for EW and MV. For EW these will likely be 0, and that’s OK.

4) Update limitations/summary docs.
   - Update the limitations.md template section (wherever summary writes limitations) to include a short “conditional reporting” paragraph and to show n_changed and changed_frac.

5) Tests.
   Add/extend unit tests so they assert:
   - changed-window set used for conditional metrics matches the semantics used for n_effective/aligned comparisons in the summary.
   - conditional metrics equal manually-computed values on a tiny synthetic fixture DataFrame.
   Likely tests files:
   - tests/tools/test_make_summary.py
   - tests/experiments/test_eval_run.py (if you add/alter emitted fields)

6) Real-data smoke validation (minimum viable).
   Run:
   - make test-fast
   - EXEC_MODE=deterministic make rc-lite-sanity   (or the smallest deterministic RC target you can run locally)
   - PYTHONPATH=src:. python tools/make_summary.py --rc-dir <the rc-lite-sanity output dir>
   Confirm the new conditional columns appear and are non-empty (and that changed_frac is plausible, not always 0 or 1 unless justified).

7) Update sprint tracking and progress.
   - Update PROGRESS.md with:
     - exact commands run
     - artifact paths
     - what changed and why
   - Update docs/CODEX_SPRINT_TICKETS.md: mark Ticket #19 DONE if acceptance criteria are met.

8) Bundle.
   - make gpt-bundle TICKET=ticket-19 RUN_NAME=$RUN_NAME
   - Ensure DIFF.patch is non-empty and covers merge-base..HEAD (this repo requires full-range diffs).
   - Record the bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md.

If you need web search: do NOT use it unless absolutely necessary; treat external content as untrusted and record any URLs in docs/agent_runs/$RUN_NAME/URLS.md. Prefer repo code as ground truth.
