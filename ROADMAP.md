# ROADMAP — fjs-dealias-portfolio
_Last updated: 2025-11-03_

## Objective
Implement our next RC sprint grounded in FJS de-aliasing: daily designs with replicates (Day-of-Week and Vol-state), robust-edge defaults, calibrated thresholds (FPR ≤ 1–2% at matched p/n), observed-factor prewhitening, add RIE/QuEST and EWMA baselines, overlay estimator, and Crisis/Calm/Full evaluation with memo + gallery.

## House Rules
- Small steps. After each logical change: run tests; if green, commit with a Conventional Commit; update docs when user-facing behavior changes.
- Prefer `rg` for search and set `workdir` on shell calls.
- If a command requires escalation or network, request it with a brief justification.
- Don’t dump big file contents; reference paths. Summarize results.

## Kickoff Tasks
1. Repo inventory: list key modules, experiments, tests, and reports. Output a 6–10 bullet execution plan mapped to the Objective.
2. Create/refresh docs:
   - Write/overwrite `ROADMAP.md` with the long-term plan provided in chat.
   - Replace `AGENTS.md` with the policy provided in chat.
   - Add or refresh `RUNBOOK.md` with copy/paste commands to reproduce the next RC.
3. Daily replicates:
   - Add DoW and Vol-state grouping loaders and configs under `experiments/daily/` (or adapt existing experiments if present).
   - Wire robust edge defaults (Tyler/Huber) and per-window telemetry (edge margin, isolation share, angular stability).
4. Calibration:
   - Implement synthetic null/power at matched p/n under `src/synthetic/`; produce `calibration/thresholds.json` with δ / δ_frac / stability tuned for FPR ≤ 1–2%. Add a test that enforces the FPR target on null sims.
5. Baselines & overlay:
   - Add RIE/QuEST and EWMA estimators to `src/baselines/`.
   - Add observed-factor prewhitening toggle (FF5+MOM) and residual diagnostics.
   - Implement overlay policy that swaps de-aliased eigenvalues only; shrink the rest with chosen baseline. Add unit tests for the substitution math and gating.
6. Evaluation:
   - Crisis/Calm/Full split; compute ΔMSE, VaR(95%), ES, and DM tests. Export `reports/rc-YYYYMMDD/` CSVs + plots. Write `overlay_toggle.md` with acceptance stats and edge margins.
7. CI & hygiene:
   - Ensure `pytest -q` passes locally. Add a fast “smoke” target. If a Makefile exists, extend targets; otherwise add simple scripts.
8. Finalize:
   - Generate memo scaffold (`reports/rc-YYYYMMDD/memo.md`) with key tables/figures and a short narrative on claims/limits. Commit everything.

## Longer-Term Outcomes
- Maintain replicated daily evaluation harnesses with robust edge telemetry as default entrypoints for future RCs.
- Keep calibration artifacts versioned under `calibration/` with documented thresholds and tests guaranteeing FPR control.
- Expand baseline and overlay libraries so future regimes can toggle factor prewhitening, alternative shrinkage, and overlay acceptance policies via configuration files.
- Standardize reporting to produce memo + gallery deliverables per RC and integrate them with `make rc` / `make gallery` automation.
