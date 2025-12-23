Implement **Ticket #17 — Nested calibration grid coverage** in repo `fjs-dealias-portfolio`.

CRITICAL: Follow `AGENTS.md` (stop-the-line rules) and `docs/DOCS_AND_LOGGING_SYSTEM.md` (logging contract). Do not create “fake fixes” (e.g., mapping missing (p,T) to a random existing cell without validating null-FPR).

Do NOT write a long upfront plan. Start by inspecting current behavior, then implement, test, and document end-to-end.

### Branch + run log (required)
1) Create a feature branch:
   - `git checkout -b codex/ticket-17-nested-calibration-coverage`

2) Set:
   - `RUN_NAME=$(date +%Y%m%d_%H%M%S)_ticket-17_nested-calibration-coverage`
   - Create `docs/agent_runs/$RUN_NAME/` and populate (REQUIRED):
     - `PROMPT.md` (paste this prompt verbatim)
     - `COMMANDS.md` (every command you run, in order)
     - `RESULTS.md` (what changed + artifact paths + any failures)
     - `TESTS.md` (exact tests run + pass/fail)
     - `META.md` (git SHA before/after, branch, dirty-at-start, dataset ids/hashes used, config hashes)

3) Commit in small logical chunks. Every commit body MUST include:
   - `Tests: <exact commands>`

### Acceptance criteria (must satisfy all)
From `docs/CODEX_SPRINT_TICKETS.md` Ticket #17:
- `make run:equity_nested_smoke_tiny` produces windows that do NOT skip with `calibration_missing_p_T`.
- `calibration/nested_edge_delta_thresholds.json` includes audit metadata (run_name, timestamp, git_sha, config_hash) and thresholds for newly required grid cells.
- Synthetic nested null-FPR at the operating point remains ≤ target (2% is fine) for newly added (p,T) cells.

### Work steps (do these, in this order)
A) Reproduce current failure (baseline evidence)
1) Run and record:
   - `make test-fast` (only if already quick; otherwise do later after code edits)
   - `EXEC_MODE=deterministic make run:equity_nested_smoke_tiny`
2) Locate outputs under `experiments/equity_panel/outputs_nested_smoke_tiny/...`
3) Extract the observed (p,T) that cause `calibration_missing_p_T` (expect p≈188, T∈{70,80}).
4) In `docs/agent_runs/$RUN_NAME/RESULTS.md`, record:
   - the pre-fix skip_reason histogram / summary showing `calibration_missing_p_T`
   - the observed (p,T) pairs

B) Understand current calibration schema + lookup strictness
1) Inspect:
   - `calibration/nested_edge_delta_thresholds.json` (existing grid + metadata)
   - `src/fjs/gating.py` (how nested looks up thresholds; how p/T are computed/bucketed)
   - `experiments/synthetic/nested_killtest.py` and `experiments/synthetic/config.nested.killtest.yaml` (how thresholds are generated)
2) Determine precisely why (p≈188, T∈{70,80}) is missing:
   - Is p binned? Are you rounding? Is T “effective T” vs “window weeks” mismatch?
   - Fix the *root cause*, not the symptom.

C) Extend calibration coverage (NO fake fixes)
Preferred approach: **actually calibrate** the missing cells.
1) Update the nested killtest config and/or generator so it can generate thresholds for the missing (p,T) cells.
   - Add cells for p around 188 (exact 188 if supported; otherwise an explicit, documented binning rule that guarantees the real-data p maps to a calibrated cell).
   - Add T={70,80} (or the exact T definition used in gating).
2) Run:
   - `python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/$RUN_NAME --calibration-out calibration/nested_edge_delta_thresholds.json`
3) Ensure the calibration JSON is updated in-place and includes audit metadata:
   - run_name, timestamp, git sha, config hash (sha256 of the written calibration file, or whatever the repo standard is)
4) Validate null-FPR for the new (p,T) cells at the operating point:
   - If the harness is configurable, choose trials high enough that a ≤2% target is meaningful (document exact trials used).
   - Record null-FPR table(s) in `reports/synthetic/nested_killtest/$RUN_NAME/` and summarize in RESULTS.md.

If you implement any fallback/approximation (e.g., nearest-neighbor):
- It MUST be explicit (logged reason code) and MUST come with synthetic evidence that null-FPR is still controlled.
- Do not silently map.

D) Update tests (fast + deterministic)
Add/extend tests to prevent regression:
- `tests/synthetic/test_calibration.py`:
  - asserts calibration file has coverage for the observed (p,T) cells OR that lookup returns a calibrated operating point without “missing”.
- `tests/test_threshold_eval.py`:
  - ensure nested lookup does not return `calibration_missing_p_T` for those (p,T) pairs.

Run (REQUIRED):
- `make test-fast`

E) Re-run real-data tiny smoke (post-fix)
- `EXEC_MODE=deterministic make run:equity_nested_smoke_tiny`
Acceptance check:
- The outputs must show **no** `calibration_missing_p_T` skips.
- If windows still skip, the reason must be something else and must be explicitly attributed.

F) Update docs + provenance (required)
1) `PROGRESS.md`:
   - Add a Ticket-17 entry with branch, SHAs, exact commands, tests, and artifact paths:
     - `reports/synthetic/nested_killtest/$RUN_NAME/`
     - `experiments/equity_panel/outputs_nested_smoke_tiny/...`
2) `project_state/KNOWN_ISSUES.md`:
   - Remove or revise the “nested calibration grid lacks p≈188 coverage” issue once fixed.
3) If nested smoke is now unblocked and yields meaningful new validity info:
   - Update `project_state/CURRENT_RESULTS.md` with a concise entry (date + what changed).

G) Finish cleanly + bundle for review
1) Ensure `git status` is clean.
2) Save a patch:
   - `git diff > docs/agent_runs/$RUN_NAME/DIFF.patch`
3) Generate a GPT bundle:
   - `make gpt-bundle TICKET=ticket-17 RUN_NAME=$RUN_NAME`
4) In `docs/agent_runs/$RUN_NAME/RESULTS.md`, record the bundle path and (recommended) `unzip -l` listing.

DONE means:
- acceptance criteria satisfied,
- tests green,
- run log complete,
- docs updated,
- no silent fallback introduced.
