You are working in repo fjs-dealias-portfolio. Follow AGENTS.md exactly (stop-the-line rules are binding).

Ticket: ticket-25
Goal: Make inject-spike scientifically valid for MANOVA by adding component-aware injection modes (between-group vs within-group vs total), then run a minimal real-data smoke to see whether WEEK design responds when we inject the “between-group” component that the theory targets.

Hard constraints:
- Do NOT disable detection/gating or “force accept/reject” anything.
- No silent fallbacks: missing configs/paths must fail loudly.
- Keep changes auditable: small commits, tests in commit body, run logs, PROGRESS.md update, and bundle at end.
- Do not use web search for this ticket.

Step 0 — Branch + baseline sanity
1) Create a feature branch: codex/ticket-25_inject-component-modes
2) Inspect current inject pipeline:
   - experiments/eval/inject_spike.py (especially injection construction + window grouping)
   - src/eval/grouping.py (or wherever groups_for_design lives)
   - src/fjs/dealias.py / src/fjs/overlay.py diagnostics expectations
3) Confirm current CLI + outputs schema and the existing tests under tests/experiments/test_inject_spike.py.

Step 1 — Implement component-aware injection (main work)
Add a new CLI flag to experiments/eval/inject_spike.py:
- --inject-mode {total,between,within}
Default must be "total" to preserve prior behavior.

Define semantics (must be deterministic given --seed and per-window basis):
- total: current behavior (row-wise scalar series z_t applied to all rows), standardized to mean 0 / std 1 over rows.
- between: group-constant injection:
    For each group g, sample u_g ~ N(0,1), assign z_t = u_{group[t]} for rows in that group.
    Standardize z over rows (mean 0, std 1) so mu is comparable across windows/modes.
    This approximates injecting a rank-1 spike into the BETWEEN-group variance component.
- within: within-group-only injection:
    Sample z_t i.i.d. then demean within each group: z_t <- z_t - mean(z within group of t),
    then standardize globally (mean 0, std 1).
    This removes any between-group mean component by construction.

Implementation details:
- Reuse the SAME per-window v (asset direction) basis already used, and reuse existing seeding conventions.
- Update run.json metadata to include inject_mode and any relevant stats (e.g., group counts, n_groups, reps_per_group, z_mean/std checks).
- Update outputs (curve.csv/windows_detail.csv/gating_reasons.csv) to include inject_mode so plots/tables are not ambiguous.

Step 2 — Tests (unit-level “synthetic”, minimal but meaningful)
Update/add unit tests in tests/experiments/test_inject_spike.py to validate injection semantics without relying on large data:
- Construct a tiny panel X with known group labels (e.g., 3 groups × 4 reps each, p small).
- Apply injection in each mode with a fixed seed and a simple v (or extracted from code).
Assertions:
  - between-mode: group means along v must differ across groups (variance of group means > 0).
  - within-mode: group means along v must be ~0 (within numerical tolerance) after injection.
  - total-mode: should generally have nonzero group means but not forced; at minimum it should preserve mean/std standardization of z.
Also keep existing tests passing (determinism, output schema).

Run:
- python -m pytest tests/experiments/test_inject_spike.py -q
- make test-fast
Record results in the run log and commit bodies.

Step 3 — Minimal real-data smoke (fixture data) to validate “between” responds
Run a SMALL smoke on the repo’s fixture data to keep runtime reasonable:
- Use reports/fixtures/returns_daily_small.csv and reports/fixtures/ff5mom_daily_small.csv
- group-design week, use-factor-prewhiten 1
- Keep max-windows small (e.g., 20) and a short mu grid (0, 12, 24)
- Use deterministic sampling settings

Example (adjust only if paths/options differ in this repo):
  RUN_ID=20251226_ticket25_week_between_smoke
  PYTHONPATH=src:. python experiments/eval/inject_spike.py \
    --returns-csv reports/fixtures/returns_daily_small.csv \
    --factors-csv reports/fixtures/ff5mom_daily_small.csv \
    --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 \
    --assets-top 25 \
    --config experiments/eval/config.yaml \
    --thresholds experiments/eval/thresholds.json \
    --group-design week --use-factor-prewhiten 1 \
    --mu-grid 0,12,24 \
    --inject-mode between \
    --max-windows 20 --window-sampling uniform --window-sampling-seed 7 \
    --seed 7 \
    --out reports/inject_spike

Acceptance check (be explicit in RESULTS.md):
- The run must complete and write run.json + resolved_config.json.
- curve.csv must include baseline (mu=0) and injected rows.
- We are NOT claiming “good results”, but we NEED signal:
    At mu=24, either detection_rate > 0 OR raw_outliers_found > 0 in windows_detail for a nontrivial fraction of injected windows.
  If it is still flat-zero, summarize exactly which guard buckets dominate now and what that implies.

Step 4 — Documentation + audit trail (mandatory)
1) Create run log: docs/agent_runs/<RUN_NAME>/ with PROMPT/COMMANDS/RESULTS/TESTS/META.
   - In COMMANDS.md include exact commands (no ellipses).
   - In RESULTS.md include the acceptance check outcome and the key tables (detection/acceptance by mu; top gating reasons).
2) Copy SMALL key artifacts into docs/agent_runs/<RUN_NAME>/artifacts/:
   - curve*.csv, curve*.png
   - gating_reasons*.csv
   - windows_detail*.csv (or a reduced version if huge)
   - run.json and resolved_config.json (copies) for auditability
3) Update PROGRESS.md with:
   - branch/run name, git sha
   - commands
   - artifact paths (reports dir + run log dir)

Step 5 — Finish clean
- Ensure git status is clean.
- Make small logical commits; each commit body must include: Tests: <commands>.
- Generate bundle and record it in RESULTS.md:
    make gpt-bundle TICKET=ticket-25 RUN_NAME=<RUN_NAME>
  Include the resulting bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md.

Stop when:
- tests pass,
- run log complete,
- PROGRESS.md updated,
- bundle generated.
