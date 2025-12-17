# 2025-12-08 Codex Sprint 01 — Nested detection & rc-lite sanity
- Time: 2025-12-08 22:20 CET
- Agent: Codex
- Sprint title: "Nested detection & rc-lite sanity"
- Sprint goal: Stabilize nested detection (avoid 0% accepted) and stand up a reliable `rc-lite-sanity` path on WRDS data that can run overnight with reports.

## Planned outcomes (1–2 week slice)
1) Implement `make rc-lite-sanity` (or equivalent) that runs a small real-data RC (DoW + vol-state, optional nested), triggers evaluation, and writes reports/figures under timestamped rc folders.
2) Diagnose and improve nested design coverage so smoke/nested runs have non-zero accepted detections while keeping synthetic null FPR near target.
3) Verify fast tests and small real-data runs on Hetzner (`make test-fast`, `make rc-lite-sanity`) pass end-to-end.
4) Add/extend nested diagnostics (skip reasons, per-year/replicate counts) to understand coverage gaps.
5) Update project docs (`PROJECT_STATE/EXPERIMENTS.md`, `ROADMAP.md` or similar) and `CHANGELOG.md` to reflect rc-lite-sanity and nested tuning status.

## Commands I expect to run
- make test-fast
- make rc-lite-sanity
- python -m experiments.equity_panel.run --config <smoke/rc-lite/nested config> [overrides]
- python -m experiments.eval.run --config <matching eval config>
- make sweep:acceptance (nested-focused, reduced trials if needed)

## Context notes
- AGENTS.md: prefer small, reviewable patches; log all steps in this file; do not touch WRDS raw data or delete outputs; use timestamped output dirs.
- LONG_TERM_PLAN (2025-12-08): short-term focus is exactly nested coverage (target 2–6% acceptance) and a reproducible rc-lite-sanity pipeline producing metrics + brief/gallery.
- PROJECT_STATE/KNOWN_ISSUES: nested coverage currently fragile (0% acceptance risk); crisis runs show overlay can harm; cache staleness caution; PSD clipping may mask instability.
- PROJECT_STATE/PIPELINE_FLOW: rc-lite/rc pipelines rely on experiments/equity_panel/run.py with calibrated gating and per-window diagnostics; reporting via tools/build_*; resume/cache keys matter.
- CONFIG_REFERENCE: equity_panel configs include gating params (delta_frac, q_max, isolation, stability), partial_week_policy, nested_replicates; gating can be calibrated via calibration/edge_delta_thresholds.json.
- EXPERIMENTS.md: small configs exist (config.smoke.yaml, config.rc-lite.yaml?) plus nested configs; ablation smoke sometimes timing out; daily eval harness exists but rc-lite path may need wiring.

## Progress log
- Setup: created `.venv`, `pip install -e .[dev]`.
- Tests: `. .venv/bin/activate && make test-fast` (pass, 65 tests).
- rc-lite-sanity iterations: initial make target too slow and overwrote rc dirs; rewired target to use timestamped `reports/rc-YYYYMMDD-sanity-<stamp>/`, direct daily eval calls (DoW/vol, 2023H1, 50 assets) plus weekly DoW + nested smoke, new summarizer `tools/summarize_rc_sanity.py`.
- Final rc-lite-sanity run (success): `reports/rc-20251208-sanity-20251209_001356/` with daily detection ~5.5% (DoW) / 5.2% (vol); weekly DoW detection 0% (4 windows); weekly nested detection 0% (10 windows, p≈188, T≈60–80). Summary/regime CSV written.
- Nested diagnostics/tuning: added per-window nested counts (years/weeks/replicates) and nested preparation logging to detection_summary/summary; relaxed nested smoke config (delta_frac 0.008, eta 0.2, nonisolated min 0.015) and shortened span (2022-01–2023-06, window 52). Standalone nested rerun (`python3 experiments/equity_panel/run.py ... --output-dir experiments/equity_panel/outputs_nested_tune_test`) still 0% accept with no isolated spikes; calibration missing for p≈188.
- Added new rc-lite-sanity summary tool and updated Makefile; nested diagnostics columns propagate to detection_summary; summary now captures detection/accept rates and nested scope.

## Current status / next steps
- rc-lite-sanity now reproducible and time-bounded (~13 min) with timestamped outputs; daily slices OK, weekly DoW/nested still zero accepts.
- Nested coverage still 0% even after relaxed gating; need further tuning/calibration (e.g., delta_frac grid for p~190, revisit dealias_search diagnostics).
- Consider additional nested diagnostics (reason codes when detections empty) and a smaller synthetic/null sweep focused on nested parameters.
