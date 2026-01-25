## Done
- 2026-01-25: FJS-TKT-024 remove .gitignore.append (agentic ignores already in .gitignore). Tests: `. .venv/bin/activate && make test-fast && python3 tools/agentic/project_state_refresh.py --zip && python3 tools/agentic/repo_snapshot.py && git status --porcelain`. Artifacts: `docs/agent_runs/20260125_234152_ticket-024_gitignore-agentic/`, `docs/_bundles/project_state_20260125_224343.zip`, `docs/_generated/repo_snapshot.md`.
- 2026-01-24: FJS-TKT-022 weekly smoke diagnostics hardening + regression assertions. Tests: `make test-fast`, `make run:equity_nested_smoke_tiny`.
- 2026-01-25: FJS-TKT-023 weekly runner exception hardening + detection_summary assertions. Tests: `make test-fast`, `make run:equity_nested_smoke_tiny`. Artifacts: `docs/agent_runs/20260125_012533_ticket-23_finish-weekly-acceptance/`, `experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`.

## 2025-12-23T22:56Z — ticket-18 injection sensitivity (real windows)
- **Branch/Run**: `codex/ticket-18-inject-spike-sensitivity` (RUN_NAME=`20251223_222840_ticket-18_inject-spike-sensitivity`), git sha `22523fc301aa7228193bf135ae9615974cb631c0`.
- **Commands**: `make test-fast`; `RC_ASSETS_TOP=50 RC_WINDOW=60 RC_HORIZON=10 RC_START=2024-01-01 RC_END=2024-03-31 make inject-spike` (failed: insufficient observations); `RC_ASSETS_TOP=50 RC_WINDOW=60 RC_HORIZON=10 RC_START=2024-01-01 RC_END=2024-06-30 make inject-spike` (interrupted); `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-03-31 --assets-top 30 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3 --seed 7 --out reports/inject_spike`; `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-03-31 --assets-top 30 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3,6,9 --seed 7 --out reports/inject_spike`; `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-09-30 --assets-top 30 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3,6,9,12,15 --seed 7 --out reports/inject_spike` (interrupted); `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3,6,9,12,15 --seed 7 --out reports/inject_spike`; `RC_ASSETS_TOP=25 RC_WINDOW=40 RC_HORIZON=5 RC_START=2024-01-01 RC_END=2024-06-30 make inject-spike`.
- **Changes**: `inject_spike.py` now emits timestamped `reports/inject_spike/<RUN_ID>/` with `curve.csv`, plot, `run.json`, `resolved_config.json`, and `selected_windows.csv`; detection/acceptance rates (pre/post gate) recorded with fixed per-window injection basis and skip-reason histogram; Makefile inject-spike uses `RC_ASSETS_TOP` and default output root `reports/inject_spike`.
- **Results**: make target run (`reports/inject_spike/20251225_213525/`, 80 windows) produced curve/plot/run.json; baseline detection/acceptance = 0.0 and μ=3/4/5 detection/acceptance = 0.0; `n_detected=0` implies pre-gate drought. Larger slice run (`reports/inject_spike/20251224_051700/`) with μ=3/6/9/12/15 is also flat zero; earlier small run (`reports/inject_spike/20251223_225141/`) matches the zero-response pattern.
- **Tests**: `make test-fast` (73 passed, 168 deselected, 1 warning: PytestConfigWarning unknown timeout option).
- **Artifacts**: run log `docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/`; outputs `reports/inject_spike/20251225_213525/` (curve.csv/curve.png/run.json/resolved_config.json/selected_windows.csv), `reports/inject_spike/20251224_051700/`, and `reports/inject_spike/20251223_225141/`; partial runs `reports/inject_spike/20251223_224624/` (failed), `reports/inject_spike/20251223_224638/` (interrupted), and `reports/inject_spike/20251224_051229/` (interrupted).

## 2025-12-23T21:22Z — ticket-22 gpt-bundle range diff + bundle meta
- **Branch/Run**: `codex/ticket-22-gpt-bundle-range-diff` (RUN_NAME=`20251223_204129_ticket-22_gpt-bundle-range-diff`), git sha `458932062b157a2697458269d855025aaddf4343`.
- **Commands**: `make test-fast`; `BUNDLE_STAMP=20251223_214500 make gpt-bundle TICKET=ticket-22 RUN_NAME=20251223_204129_ticket-22_gpt-bundle-range-diff`; `unzip -l docs/gpt_bundles/20251223_214500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip`; `unzip -p docs/gpt_bundles/20251223_214500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip BUNDLE_META.md`; `unzip -p docs/gpt_bundles/20251223_214500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | head -n 30`; `unzip -p docs/gpt_bundles/20251223_214500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | rg -n "tools/gpt_bundle.py"`; `unzip -p docs/gpt_bundles/20251223_214500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | rg -n "AGENTS.md"`; `BUNDLE_STAMP=20251223_214700 make gpt-bundle TICKET=ticket-22 RUN_NAME=20251223_204129_ticket-22_gpt-bundle-range-diff`.
- **Changes**: gpt-bundle now emits merge-base range diffs with base auto-detect/BUNDLE_BASE override, writes `BUNDLE_META.md`, and fails loud on missing base refs or empty diffs; added range-diff unit coverage + base-ref failure test; docs updated to require full ticket delta and bundle meta.
- **Tests**: `make test-fast` (71 passed, 168 deselected, 1 warning: PytestConfigWarning unknown timeout option).
- **Artifacts**: run log `docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/`; bundle (manual smoke) `docs/gpt_bundles/20251223_214500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip`; bundle (reviewer) `docs/gpt_bundles/20251223_214700_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip`.

## 2025-12-23T19:50Z — ticket-21 gpt-bundle diff auditability
- **Branch/Run**: `codex/ticket-21-gpt-bundle-diff` (RUN_NAME=`20251223_203756_ticket-21_gpt-bundle-diff`), git sha `001908dab818701f53ba90016bbca48e8087774c`.
- **Commands**: `make test-fast` (x3); `make gpt-bundle TICKET=ticket-21 RUN_NAME=20251223_203756_ticket-21_gpt-bundle-diff`; `unzip -l docs/gpt_bundles/20251223_204539_ticket-21_20251223_203756_ticket-21_gpt-bundle-diff.zip`; `unzip -p docs/gpt_bundles/20251223_204539_ticket-21_20251223_203756_ticket-21_gpt-bundle-diff.zip DIFF.patch | head -n 20`; `unzip -p docs/gpt_bundles/20251223_204539_ticket-21_20251223_203756_ticket-21_gpt-bundle-diff.zip DIFF.patch | wc -c`; `BUNDLE_STAMP=20251223_205150 make gpt-bundle TICKET=ticket-21 RUN_NAME=20251223_203756_ticket-21_gpt-bundle-diff`.
- **Changes**: gpt-bundle now emits DIFF.patch via git show helper (non-empty on clean trees), validates required run log files, and supports a fixed bundle stamp; added regression test for diff generation; docs updated to codify bundle auditability.
- **Results**: manual smoke bundle DIFF.patch non-empty; final bundle path recorded below.
- **Tests**: `make test-fast` (70 passed, 168 deselected, 1 warning: PytestConfigWarning unknown timeout option).
- **Artifacts**: run log `docs/agent_runs/20251223_203756_ticket-21_gpt-bundle-diff/`; bundle `docs/gpt_bundles/20251223_205150_ticket-21_20251223_203756_ticket-21_gpt-bundle-diff.zip`.

## 2025-12-23T18:26Z — ticket-17 nested calibration coverage
- **Branch/Run**: `codex/ticket-17-nested-calibration-coverage` (RUN_NAME=`20251223_180034_ticket-17_nested-calibration-coverage`), git sha `b2221e8241edee8cfcf76fb454d8b1a8a51f8add`.
- **Commands**: `make test-fast`; `EXEC_MODE=deterministic make run:equity_nested_smoke_tiny`; `python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage --calibration-out calibration/nested_edge_delta_thresholds.json`; `make test-fast`.
- **Changes**: nested killtest now supports multi-asset grids (`n_assets_options`) and emits schema-1 calibration JSON with audit metadata + design_thresholds; killtest writes `resolved_config.json` + hashes in `run.json`; config updated to include p=188; nested calibration JSON refreshed with `188x{60,70,80}` and metadata; tests added to assert nested calibration coverage and lookup for p=188.
- **Results**: nested killtest null FPR 0/220 for p=188 (Wilson hi 0.017); deterministic tiny nested smoke now skips due to `instability_in_a_neighborhood`/`no_isolated_spike` (no `calibration_missing_p_T`).
- **Tests**: `make test-fast` (pass).
- **Artifacts**: `reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/`; `experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`; run log `docs/agent_runs/20251223_180034_ticket-17_nested-calibration-coverage/`.

## 2025-12-22T21:06Z — project_state rebuild (doc-only)
- **Branch/Run**: `chore/project_state_refresh` (RUN_NAME=`20251222_205400_project_state_rebuild`), git sha `4dfb862085cd47cba810cedc4ea1bc5489ce0bfe`.
- **Commands**: `python3 tools/generate_project_state.py`; `python3 - <<'PY'` (emit FUNCTION_INDEX/DEPENDENCY_GRAPH); `python3 - <<'PY'` (write project_state docs); `zip -r docs/gpt_bundles/project_state_20251222_205400_4dfb862.zip ...`.
- **Changes**: regenerated `project_state/_generated` artifacts, rewrote all `project_state/*.md` with updated module inventory and run summaries; updated `tools/generate_project_state.py` to scope AST indexing to src/experiments/tools and include signatures/bases; noted missing `experiments/eval/config.paper_v1.yaml` in Known Issues.
- **Tests**: Not run (documentation-only).
- **Artifacts**: run log `docs/agent_runs/20251222_205400_project_state_rebuild/`; bundle `docs/gpt_bundles/project_state_20251222_205400_4dfb862.zip`.

## 2025-12-21T06:21Z — WRDS lake inventory doc
- **Branch/Run**: `feat/ticket-02-stop-eval-contamination` (RUN_NAME=`20251221_072005_ticket-02_wrds-lake-doc`), git sha `27a3afcb4770ad24800efd30af1a5ee4451f806f`.
- **Commands**: `make test-fast`.
- **Changes**: documented external WRDS lake structure (user mirror at `/Volumes/Storage/Data`) in `docs/DOCS_AND_LOGGING_SYSTEM.md` for future refresh provenance; noted that mirror is local-only and not visible in CI.
- **Artifacts**: run log `docs/agent_runs/20251221_072005_ticket-02_wrds-lake-doc/`.

## 2025-12-21T06:00Z — data mount documentation (external WRDS mirror)
- **Branch/Run**: `feat/ticket-02-stop-eval-contamination` (RUN_NAME=`20251221_065852_ticket-00_data-mount`), git sha `8ac9c14cd0e134aee8bf7a803891a0c476250a95`.
- **Actions**: documented the requirement to log external data mirror paths in `docs/DOCS_AND_LOGGING_SYSTEM.md`; attempted to inspect `/Volumes/Storage/Data` but the path is not visible in this environment.
- **Notes**: user reports WRDS data mirrored at `/Volumes/Storage/Data`; ensure the repo points to it via symlink/bind mount under `data/` and record verification status in the run log.
- **Artifacts**: run log `docs/agent_runs/20251221_065852_ticket-00_data-mount/`.

## 2025-12-21T04:15Z — ticket-02 stop eval contamination (caps + solver)
- **Branch/Run**: `feat/ticket-02-stop-eval-contamination` (RUN_NAME=`20251221_042859_ticket-02_stop-eval-contamination`), git sha `6df53dd0686292b98d709bd98f1894f14a076bed`.
- **Commands**: `make test-fast`; `EXEC_MODE=deterministic make rc-lite-sanity` (timed out at 120s/300s/600s; dow leg completed; vol leg completed manually; weekly legs not run); `EXEC_MODE=deterministic PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py --returns-csv data/returns_daily.csv --window 60 --horizon 10 --start 2023-01-01 --end 2023-06-30 --assets-top 50 --group-design vol --group-min-count 3 --group-min-replicates 6 --min-reps-vol 6 --edge-mode tyler --shrinker oas --prewhiten ff5mom --overlay-delta 0.05 --coarse-candidate 1 --gate-mode soft --gate-accept-nonisolated --gate-delta-calibration calibration/edge_delta_thresholds.json --gate-delta-frac-min 0.015 --q-max 2 --mv-gamma 1e-4 --mv-box 0.0,0.1 --mv-turnover-bps 5 --mv-condition-cap 1000000 --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv --out reports/rc-20251221-sanity-20251221_045550/vol-tyler`; `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251221-sanity-20251221_045550`; `EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --window 40 --horizon 5 --out reports/smoke_cap_test --assets-top 20 --group-design dow --shrinker rie --prewhiten off --use-factor-prewhiten 0 --q-max 2 --mv-box-lo -0.25 --mv-box-hi 0.25 --mv-turnover-bps 0.0 --mv-condition-cap 1000000 --max-windows 5 --min-comparison-windows 3 --seed 123 --workers 1`; `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/smoke_cap_test`.
- **Results**: `tools/make_summary.py` now hard-excludes `cap_active=true` runs (and MV skip-on-missing-solver runs) from headline tables; `limitations.md` gains explicit sections listing excluded capped runs + cap_sources and smoke-only MV-skip runs; `experiments/eval/run.py` always writes cap metadata into `run.json` even on empty metrics. Capped smoke (`reports/smoke_cap_test/`) yields empty summary tables with `limitations.md` showing `cap_sources: max_windows, window_coverage`. RC-lite sanity (`reports/rc-20251221-sanity-20251221_045550/`) is capped via `date_truncation`, so summary tables are empty and limitations list both dow/vol legs as excluded.\n- **Artifacts**: Run log `docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/`; RC-lite outputs `reports/rc-20251221-sanity-20251221_045550/`; capped smoke `reports/smoke_cap_test/`.

## 2025-12-21T01:36Z — ticket-01 overlay forensics (daily RC)
- **Branch/Run**: `feat/ticket-01-overlay-forensics` (RUN_NAME=`20251221_015106_ticket-01_overlay-forensics`), git sha `d3d1ac271fb7b3e0246b1e6a292dc6062fa1d062`.
- **Commands**: `make test-fast` (failed once due to indentation, reran and passed after fix); `EXEC_MODE=deterministic make rc-lite-sanity` (timed out; completed vol leg manually); `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251221-sanity-20251221_021657`.
- **Results**: `summary/overlay_forensics.csv` emitted for `reports/rc-20251221-sanity-20251221_021657/` (214 changed-window rows); `summary/limitations.md` now points to overlay_forensics for ΔMSE/ΔQLIKE attribution; `summary/completeness.json` reports `cap_active=false`.
- **Artifacts**: run log `docs/agent_runs/20251221_015106_ticket-01_overlay-forensics/`; RC-lite outputs `reports/rc-20251221-sanity-20251221_021657/`.

## 2025-12-20T23:26Z — project_state rebuild (doc-only)
- **Branch/Run**: `chore/project_state_refresh` (RUN_NAME=`20251220_232502_project_state_rebuild`), git sha `a7d76d8cf7f5fe4c9765c335530064170a0ca87a`.
- **Commands**: `python tools/generate_project_state.py`; `python - <<'PY'` (rebuild project_state docs + ASCII normalization + header refresh).
- **Changes**: regenerated `project_state/_generated/{repo_inventory.json,symbol_index.json,import_graph.json,make_targets.txt}`, rewrote all `project_state/*.md`, updated `tools/generate_project_state.py` for module naming + scope.
- **Artifacts**: run log `docs/agent_runs/20251220_232502_project_state_rebuild/`; bundle `docs/gpt_bundles/project_state_20251220_232605_a7d76d8.zip`.
- **Tests**: `make test-fast` (69 passed, 161 deselected, 1 warning: PytestConfigWarning for unknown timeout option).

## 2025-12-20T22:47Z — ticket-09 guard attribution (guard_unknown surfacing)
- **Branch/Run**: `codex/ticket-09-weekly-gating-attribution` (RUN_NAME=`20251220_223706_ticket-09_weekly-gating-attribution`), git sha `00159178ef9f9dac4f06fc048d62d88df1bb908f`.
- **Commands**: `pytest tests/experiments/test_gating_diagnostics.py`; `make test-fast` (first attempt timed out at 10s, reran with 120s timeout and passed); `EXEC_MODE=deterministic make run:equity_smoke`.
- **Changes**: Added `guard_unknown` skip reason for unknown guard keys, included a `guard_unknown` column in gating diagnostics, and added a regression test to force actionable attribution; diagnostic_failure path still records exception type/stage/detail.
- **Results**: Equity smoke (`experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`) detection_rate=0.75 (3/4); lone rejection is `no_isolated_spike`; `guard_unknown` total=0 and `guard_other` absent; weekly_diagnostics.md shows skip_reason_primary counts + example window; diagnostic_failure not triggered in smoke (covered by tests). Remaining limitation: unknown guard path only exercised in unit test—monitor future smokes for unexpected guard keys.
- **Artifacts**: Run log `docs/agent_runs/20251220_223706_ticket-09_weekly-gating-attribution/`; smoke outputs in the path above.

## 2025-12-20T17:20Z — ticket-15 fixup for ticket-11 (eval contamination audit)
- **Branch/Run**: `codex/ticket-15-ticket11-fixup` (RUN_NAME=`20251220_171911_ticket-15_ticket11-fixup`), git sha `ffc442e3951f3dfa54759366a1107ae1f848e94b`.
- **Commands**: `pytest tests/experiments/test_eval_run.py::test_aligned_delta_and_dm_use_window_intersection tests/experiments/test_eval_run.py::test_run_evaluation_marks_comparison_valid_and_caps tests/experiments/test_eval_run.py::test_run_evaluation_delta_respects_changed_window_filter`; `make test-fast`; `EXEC_MODE=deterministic python3 -m experiments.eval.run --returns-csv data/returns_daily.csv --window 40 --horizon 5 --out reports/ticket-15-smoke-171911 --assets-top 20 --shrinker rie --use-factor-prewhiten 0 --prewhiten off --q-max 2 --mv-box-lo -0.25 --mv-box-hi 0.25 --mv-turnover-bps 0.0 --mv-condition-cap 1000000 --max-windows 5 --min-comparison-windows 3 --seed 123 --workers 1`.
- **Changes**: Δ/DM comparisons now restricted to changed-window intersections via `n_effective_*` + `comparison_valid*`; added `forced_changed_windows` hook for tests; added regression tests for aligned deltas/DM and window-cap validity; `CONFIG_REFERENCE.md` documents min-comparison-windows and aligned validity.
- **Results**: CAPPED deterministic smoke (`reports/ticket-15-smoke-171911/`, max_windows=5, assets_top=20) shows `n_effective_*` + `comparison_valid*` in `full/metrics.csv` and `full/dm.csv` (overlay row `n_effective_mse=5`, `comparison_valid=0`); `skip_stats.csv` lists skip shares by estimator/reason; `run.json` windows block flags `cap_active=true`, `cap_sources=[max_windows, window_coverage]`, `window_coverage≈0.0013` (labelled not for headline).
- **Artifacts**: Run log `docs/agent_runs/20251220_171911_ticket-15_ticket11-fixup/`; smoke outputs `reports/ticket-15-smoke-171911/`; bundle `docs/gpt_bundles/20251220_174554_ticket-15_20251220_171911_ticket-15_ticket11-fixup.zip` (contents logged in `docs/agent_runs/20251220_171911_ticket-15_ticket11-fixup/bundle_contents.txt`).

## 2025-12-20T04:20Z — ticket-14 fixup: nested calibration audit + tiny smoke (codex/ticket-14-ticket10-fixup)
- **Branch/Run**: `codex/ticket-14-ticket10-fixup` (RUN_NAME=`20251220_035705_ticket-14_ticket10-fixup`), base sha `334e86d7ff94aadce6e2c3f86149c198fd9bfdb0`.
- **Commands**: `.venv/bin/pip install -e '.[dev]'` after `python3 -m venv .venv`; `. .venv/bin/activate && make test-fast`; `. .venv/bin/activate && make run:equity_nested_smoke_tiny` (max_windows=3, deterministic, with gating diagnostics).
- **Changes**: `calibration/nested_edge_delta_thresholds.json` now embeds audit metadata (run_name, timestamp_utc, git_sha, config_hash, trials_per_scenario, operating_points) and mirrors thresholds under `design_thresholds.nested`; `lookup_calibrated_delta` is design-strict (returns None when the requested design block is absent) with regression tests; nested killtest defaults repointed to the nested calibration; equity weekly runner gained `max_windows` support (doc’d) plus a tiny nested smoke config/Make target.
- **Results**: Calibration unchanged numerically (null detections 0/220, Wilson hi=0.017, power_moderate=power_strong=1.0 at delta_frac=0.05). Tiny nested smoke (WRDS daily returns, window 52×1, cap 3 windows) produced detection_windows=0; all 3 windows skipped with `skip_reason=calibration_missing_p_T` (p=188, T∈{70,80} outside calibrated grid), guard tallies `stability_fail=3` only; delta_frac_used=0.008 (config fallback). Run cap labelled via `max_windows: 3` in `config_resolved.yaml`.
- **Artifacts**: `calibration/nested_edge_delta_thresholds.json`; smoke outputs under `experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/` (gating_diagnostics.csv, summary.json); config `experiments/equity_panel/config.nested.smoke.tiny.yaml`; run log `docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/`.

## 2025-12-20T02:23Z — ticket-10 nested null calibration (ticket-10-nested-null-fpr)
- **Branch/Run**: `ticket-10-nested-null-fpr` (RUN_NAME=`20251220_011519_ticket-10_nested-null-fpr`), git sha `e6e798288c117a188db38c4dde85cf91972921d8`.
- **Data**: synthetic nested generator (p=200, years=2, weeks 6–8, reps=5); real smoke uses WRDS daily returns + ff5mom factors via `experiments/equity_panel/config.smoke.yaml`.
- **Commands**:
  - `source .venv/bin/activate && python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr --calibration-out calibration/nested_edge_delta_thresholds.json --run-name 20251220_011519_ticket-10_nested-null-fpr --target-fpr 0.02`
  - `source .venv/bin/activate && make test-fast`
  - `source .venv/bin/activate && EXEC_MODE=deterministic make run:equity_smoke`
- **Results**:
  - Nested killtest now applies overlay-aligned gating (admissible_root, stability floor, delta bounds, q_max); null FPR 0/220 with Wilson upper bound 0.017 (<2%) and power 1.0 on moderate/strong spikes at delta_frac=0.05.
  - Design-aware calibration emitted to `calibration/nested_edge_delta_thresholds.json` (run_name/git_sha embedded) and nested configs repointed; `lookup_calibrated_delta` accepts a `design` key.
  - Tests refreshed (`make test-fast`), calibration lookup tests assert metadata; real-data smoke still passes.

## 2025-12-19T20:16Z — project_state rebuild (doc-only @ ce4c1b2)
- **Branch/Run**: `chore/project_state_refresh` (RUN_NAME=`20251219_210410_project_state_rebuild`), git sha `ce4c1b224c43028bb5388efdebbe0e8eb52e6c61`.
- **Commands**: `python3 tools/generate_project_state.py` (post-fix rerun to clean make_targets), small Python emitters for FUNCTION_INDEX/DEPENDENCY_GRAPH + markdown rewrites, `zip -r docs/gpt_bundles/project_state_20251219_211602_ce4c1b2.zip ...`.
- **Changes**: Added `tools/generate_project_state.py`; regenerated `project_state/_generated/{repo_inventory,import_graph,symbol_index,make_targets}.json/txt`; rewrote all project_state markdowns with metadata headers and refreshed CURRENT_RESULTS/KNOWN_ISSUES/ROADMAP for tickets 05/07/08.
- **Tests**: Not run (documentation-only).
- **Artifacts**: `docs/gpt_bundles/project_state_20251219_211732_ce4c1b2.zip`; run log `docs/agent_runs/20251219_210410_project_state_rebuild/`.

## 2025-12-19T20:25Z — MV solver missing-proof (ticket-08 @ a4451969)
- **Branch/Run**: `codex/ticket-08-solver-missing-proof` (RUN_NAME=`20251219_202301_ticket-08_solver-missing-proof`), git sha `a44519691f94010993176f74949485f68b9a44f0`.
- **Commands**:
  - Tests: `source .venv/bin/activate && make test-fast` (first attempt timed out at 10 s; reran successfully: 68 passed, 151 deselected).
  - Smokes: `source .venv/bin/activate && EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --out reports/eval-smoke-ticket08-proof/normal --max-windows 2 --assets-top 50 --overlay-delta 0.2 --mv-box-lo 0.0 --mv-box-hi 0.1 --mv-solver cvxpy`; `source .venv/bin/activate && FJS_FORCE_MISSING_CVXPY=1 EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --out reports/eval-smoke-ticket08-proof/missing-skip --max-windows 2 --assets-top 50 --overlay-delta 0.2 --mv-box-lo 0.0 --mv-box-hi 0.1 --mv-solver cvxpy --mv-skip-on-missing-solver`.
- **Changes**:
  - `finance/portfolios`: add `skip_reason/solver_used`, support `FJS_FORCE_MISSING_CVXPY`, remove success-shaped fallback when solver missing, allow ridge/box passthrough in `optimize_portfolio`.
  - `experiments/eval`: new `mv_solver`/`mv_skip_on_missing_solver` knobs, propagate `skipped/skip_reason/solver_status` into metrics + diagnostics, and add regression tests covering forced-missing cvxpy.
  - Docs: `project_state/CONFIG_REFERENCE.md` documents the new solver knob + env flag; run artifacts written under `reports/eval-smoke-ticket08-proof/`.
- **Results**:
  - Normal cvxpy smoke: `reports/eval-smoke-ticket08-proof/normal/metrics_detail.csv` shows MV rows with `solver_status=optimal`, `skipped=False`.
  - Forced missing (skip flag): `reports/eval-smoke-ticket08-proof/missing-skip/full/diagnostics.csv` logs `mv_skipped_share=1.0`; `metrics_detail.csv` rows carry `skipped=True`, `skip_reason=missing_solver`, `solver_status=missing_solver`.
  - Default path remains fail-loud via `MissingSolverError` (unit tests cover) with no equal-weight fallback.
  - Bundle: `docs/gpt_bundles/20251219_204908_ticket-08_20251219_202301_ticket-08_solver-missing-proof.zip`.

## 2025-12-19T19:30Z — MV solver fail-loud (ticket-08 @ 3820c1fb85)
- **Branch/Run**: `ticket-08-solver-fallback-fail-loud` (RUN_NAME=`20251219_192721_ticket-08_solver-fallback-fail-loud`), git sha `3820c1fb850968718b43e1c4a3f00aa3b6f872c0`.
- **Commands**:
  - Env/tests: `python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]`; `source .venv/bin/activate && make test-fast` (pass).
  - Smoke: `source .venv/bin/activate && EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --out reports/eval-smoke-ticket08 --max-windows 2 --assets-top 50 --overlay-delta 0.2 --mv-box-lo 0.0 --mv-box-hi 0.1`.
- **Changes**:
  - Added explicit `MissingSolverError` for cvxpy absence; `optimize_portfolio` now fails loud by default and exposes `skip_on_missing_solver` escape hatch that returns flagged, empty-weight results (no EW fallback). `OptimizationResult` carries `solver_status` and `skipped`.
  - New unit tests simulate missing solver and assert no equal-weight fallback.
  - Docs: removed silent-fallback issue from `project_state/KNOWN_ISSUES.md`; noted skip knob in `project_state/CONFIG_REFERENCE.md`.
- **Artifacts**:
  - Smoke output: `reports/eval-smoke-ticket08/`.
  - Run log: `docs/agent_runs/20251219_192721_ticket-08_solver-fallback-fail-loud/` (PROMPT/COMMANDS/RESULTS/TESTS/META).
  - Bundle: `docs/gpt_bundles/20251219_194020_ticket-08_20251219_192721_ticket-08_solver-fallback-fail-loud.zip`.

## 2025-12-19T18:02Z — weekly gating diagnostics (ticket-07 @ 2e0fd573b5)
- **Branch/Run**: `codex/ticket-07-weekly-drought-diagnostics` (RUN_NAME=`20251219_173231_ticket-07_weekly-drought-diagnostics`), git sha `2e0fd573b509173c456923ced807be5525b38df0`.
- **Commands**:
  - Tests: `source .venv/bin/activate && make test-fast`.
  - Smokes: `source .venv/bin/activate && python -m experiments.equity_panel.run --config docs/agent_runs/20251219_173231_ticket-07_weekly-drought-diagnostics/config.synthetic.yaml --gating-diagnostics --exec-mode deterministic`; `source .venv/bin/activate && python -m experiments.equity_panel.run --config experiments/equity_panel/config.smoke.yaml --gating-diagnostics --output-dir experiments/equity_panel/outputs_smoke_ticket07_20251219_173231 --exec-mode deterministic`.
  - Bundle: `make gpt-bundle TICKET=ticket-07 RUN_NAME=20251219_173231_ticket-07_weekly-drought-diagnostics`.
- **Findings**:
  - DoW weekly smoke (2023Q1, window=6, horizon=1, edge=scm) now shows detection_rate=0.75 (3/4) with a single skip_reason `no_isolated_spike`; guardrail tallies dominated by `guard_other`=1148 despite fixed delta_frac_used=0.02. Summary: `experiments/equity_panel/outputs_smoke_ticket07_20251219_173231/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/weekly_diagnostics.md`.
  - Synthetic micro smoke on generated returns (config.synthetic.yaml) records detection_rate=0 with skip_reason `diagnostic_failure` on all 6 windows; `guard_other`=18. Indicates diagnostic/guardrail “other” path still active on tiny panels.
- **Artifacts**:
  - Real run: `experiments/equity_panel/outputs_smoke_ticket07_20251219_173231/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/{gating_diagnostics.csv,weekly_diagnostics.md}`.
  - Synthetic run: `experiments/equity_panel/outputs_ticket07_synth_20251219_173231/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/{gating_diagnostics.csv,weekly_diagnostics.md}`.
  - Bundle: `docs/gpt_bundles/20251219_180641_ticket-07_20251219_173231_ticket-07_weekly-drought-diagnostics.zip` (listed in `docs/agent_runs/20251219_173231_ticket-07_weekly-drought-diagnostics/bundle_contents.txt`).

## 2025-12-19T07:37Z — gpt-bundle restore + regression guard (ticket-06 @ d6c09b0027)
- **Branch/Run**: `ticket-06-gpt-bundle-restore` (RUN_NAME=`20251219_072353_ticket-06_gpt-bundle-restore`), git sha `d6c09b0027`.
- **Commands**:
  - Env: `python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]`.
  - Tests: `source .venv/bin/activate && make test-fast` (68 passed, 144 deselected; DeprecationWarning from utcnow remains).
  - Bundle: `make gpt-bundle TICKET=ticket-06 RUN_NAME=20251219_072353_ticket-06_gpt-bundle-restore`; listing via `unzip -l docs/gpt_bundles/*ticket-06*20251219_072353_ticket-06_gpt-bundle-restore*.zip | tee docs/agent_runs/20251219_072353_ticket-06_gpt-bundle-restore/bundle_contents.txt`.
- **Changes**:
  - Added fail-loud `gpt-bundle` target (POSIX shell) emitting DIFF.patch, LAST_COMMIT.txt, required docs, and run log into `docs/gpt_bundles/<stamp>_<ticket>_<RUN_NAME>.zip`.
  - Restored required docs (`docs/PLAN_OF_RECORD.md`, `docs/DOCS_AND_LOGGING_SYSTEM.md`, `docs/CODEX_SPRINT_TICKETS.md`) describing plan-of-record and logging/bundle rules.
  - Added regression test `tests/test_gpt_bundle.py` asserting Makefile lists the gpt-bundle target and required file paths; updated .gitignore for `bundles/` and `docs/gpt_bundles/`, untracked legacy bundles/.
- **Artifacts**:
  - Bundle: `docs/gpt_bundles/20251219_074334_ticket-06_20251219_072353_ticket-06_gpt-bundle-restore.zip` (contents logged in `docs/agent_runs/20251219_072353_ticket-06_gpt-bundle-restore/bundle_contents.txt`).
  - Run log: `docs/agent_runs/20251219_072353_ticket-06_gpt-bundle-restore/` (PROMPT, COMMANDS, RESULTS, TESTS, META).

## 2025-12-19T05:17Z — rc-lite-sanity completeness hardening (ticket-05 @ 03d4c03c)
- **Branch/Run**: `ticket-05-rc-sanity-summary-hardening` (RUN_NAME=`20251219_044404_ticket-05_rc-sanity-summary-hardening`), git sha `03d4c03c`.
- **Data**: WRDS daily returns `data/returns_daily.csv` (sha256=96ac7dd3…3197) and FF5+MOM factors `data/factors/ff5mom_daily.csv` (sha256=469d44ad…908ca); verified via `tools/verify_dataset.py` inside `make rc-lite-sanity`.
- **Commands**:
  - Env/tests: `.venv` bootstrap + `pip install -e .[dev]`; `source .venv/bin/activate && make test-fast`; `source .venv/bin/activate && pytest -m unit -k "summary or summarize_rc_sanity or run_meta"`.
  - RC-lite sanity (deterministic): `source .venv/bin/activate && EXEC_MODE=deterministic make rc-lite-sanity`.
  - Summary regen with completeness: `source .venv/bin/activate && PYTHONPATH=src:. python3 tools/make_summary.py --rc-dir reports/rc-20251219-sanity-20251219_050735` and `python3 tools/summarize_rc_sanity.py --rc-dir reports/rc-20251219-sanity-20251219_050735 --dow-dir .../dow-tyler --vol-dir .../vol-tyler --weekly-dow-dir .../dow-weekly --nested-dir .../nested`.
- **Artifacts**:
  - RC root: `reports/rc-20251219-sanity-20251219_050735/` with refreshed `summary_sanity.json`, `regime.csv`, and `summary/{summary_perf.csv,summary_detection.csv,kill_criteria.json,limitations.md,completeness.json}`.
  - Weekly outputs: `experiments/equity_panel/outputs_rc-lite-20251219_20251219_050735/{dow-weekly,nested}/`.
  - Run log: `docs/agent_runs/20251219_044404_ticket-05_rc-sanity-summary-hardening/`.
- **Results**:
  - Completeness surfaced in summaries; `incomplete_runs` is empty for this drop. Aggregate includes only complete, uncapped runs.
  - Daily DoW: detection_rate≈0.055, ΔMSE(EW)=+1.24e-10, ΔMSE(MV)=+4.52e-11, overlay_effect=harmful.
  - Daily vol: detection_rate≈0.052, ΔMSE(EW)=+3.67e-11, ΔMSE(MV)=+1.24e-13, overlay_effect=harmful.
  - Weekly DoW & nested: detection_rate=0, accept_share=0 (smoke still non-detecting under current guardrails).

## 2025-11-22T00:56Z — Hetzner RC-lite + calibration refresh (git sha 3db9335)
- **Data**: WRDS daily returns `data/returns_daily.csv` (sha256=96ac7dd3…3197) + FF5+MOM factors `data/factors/ff5mom_daily.csv` (sha256=469d44ad…908ca), verified against registries.
- **Commands**:
  - Env/registry: `make setup`; `python - <<'PY' from pathlib import Path; from data.registry import assert_registered_dataset; assert_registered_dataset(Path('data/returns_daily.csv')); PY`.
  - Fast tests: `make test-fast`.
  - RC-lite eval (deterministic, capped windows): `PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py ... --group-design dow --workers 24 --max-windows 200 --out reports/rc-20251121/dow-tyler` and same for `--group-design vol --shrinker oas --gate-delta-frac-min 0.015 --q2-alignment-min-cos 0.9` (see run_manifest for exact commands); `tools/make_summary.py --rc-dir reports/rc-20251121`.
  - Artifacts: `make rc-lite EXEC_MODE=deterministic` (smoke + crisis panels), `python tools/build_memo.py --config experiments/equity_panel/config.rc.yaml`, `python tools/build_brief.py --config experiments/equity_panel/config.rc.yaml`.
  - Calibration: `HARNESS_TRIALS=800 EXEC_MODE=deterministic make sweep:acceptance`.
  - Final tests: `make test-fast`.
- **Results**:
  - RC-lite (DoW/vol, top-60, 126×21, FF5+MOM, first 200 windows): `dow-tyler` detection≈4.32%, acceptance≈4.32%, ΔMSE(EW)=+1.75e-13, ΔMSE(MV)=−2.54e-14, percent_changed≈100%; `vol-tyler` detection≈4.33%, acceptance≈4.33%, ΔMSE(EW)=−1.05e-13, ΔMSE(MV)=−8.64e-14, percent_changed≈100%. Artifacts in `reports/rc-20251121/` (`run_manifest.json`, `metrics_summary.json`, DM tables, risk/diagnostics, prewhiten telemetry).
  - Synthetic ROC sweep (SCM energy-floor): target FPR 2% with threshold ≈0.108 (Tyler FPR≈8.5% at same cut), parameters {delta=0.5, delta_frac=0.02, eps=0.02, stability_eta=0.4}; average power=1.0 at μ ∈ {4,6,8}. Figures refreshed under `reports/figures/`, `calibration_defaults.json` timestamped `2025-11-21T22:07:47Z`.
  - Memo/brief refreshed (`reports/memo.md`, `reports/brief.md`, `reports/memo_20251122_005614.md`, `reports/brief_20251122_005625.md`); gallery already up-to-date.

## 2025-11-13T04:05Z — vol acceptance tuning + paired AWS runs (feat/vol-acceptance-prewhiten@HEAD)
- **Data**: `data/returns_daily.csv` (`sha256=96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) + `data/factors/ff5mom_daily.csv` (`sha256=469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`); both re‑verified via `python tools/verify_dataset.py …`.
- **Local commands**:
  1. `python tools/verify_dataset.py data/returns_daily.csv --registry data/registry.json`
  2. `python tools/verify_dataset.py data/factors/ff5mom_daily.csv --registry data/factors/registry.json`
  3. `make test-fast` (unit suite: 65 pass / 140 deselected).
  4. `INSTANCE_DNS=… KEY_PATH=… make aws:rc-vol AWS_ARGS='USE_FACTORS=0 EXEC_MODE=deterministic RC_PROGRESS=1 RC_GATE_DELTA_FRAC_MIN=0.015 RC_VOL_MIN_REPS=10 RC_VOL_GROUP_REPS=6 Q_MAX_VOL=2 RC_VOL_ASSETS=80'`
  5. `INSTANCE_DNS=… KEY_PATH=… make aws:rc-vol AWS_ARGS='USE_FACTORS=1 EXEC_MODE=deterministic RC_PROGRESS=1 RC_GATE_DELTA_FRAC_MIN=0.015 RC_VOL_MIN_REPS=10 RC_VOL_GROUP_REPS=6 Q_MAX_VOL=2 RC_VOL_ASSETS=80'`
  6. `python tools/prewhiten_effect.py --off reports/rc-20251113/vol-off --on reports/rc-20251113/vol-ff5mom --mirror`
  7. `make gallery` and `make memo`
- **Artifacts**:
  - Run metadata: `reports/runs/20251112T232711Z/` (vol-off) and `reports/runs/20251113T014417Z/` (vol-ff5mom) with `make_*.log`, telemetry (`metrics*.json[ln]`), `run.json`, etc.
  - Bounded RC outputs (top‑80, 126×21) under `reports/rc-20251113/{vol-off,vol-ff5mom}/` including `metrics.csv`, `risk.csv`, `diagnostics.csv`, `dm.csv`, `dm_flip_only.csv`, `flip_dm.png`, and mirrored `prewhiten_effect.csv`.
  - Manifest + doc updates: `reports/rc-20251113/run_manifest.json`, refreshed `reports/memo.md` + timestamped `reports/memo_20251113_040459.md`, README/REPORT sections describing the new knobs + reporting stack.
- **Results**:
  - Acceptance (full regime) landed inside the 2–6 % band: off run `acceptance_rate=2.38 %`, FF5+MOM `acceptance_rate=2.26 %` with substitution fractions ≈1.6 % and ≈1.5 % respectively on the top‑80 universe (`reports/rc-20251113/*/full/diagnostics.csv`).
  - Flip-set telemetry: `reports/rc-20251113/vol-off/dm_flip_only.csv` shows `n_effective=110` (stats are NaN because residuals match baseline), while the prewhitened run logs `n_effective=117` with sign-test stats (`ew vs baseline`: z≈5.57, p≈9.3e‑10; `mv vs baseline`: z≈3.36, p≈9.8e‑4).
  - `reports/rc-20251113/vol-ff5mom/prewhiten_effect.csv` summarises the paired deltas: detection_rate +3.6 bps, ΔMSE(EW)=+4.1e‑11, ΔMSE(MV)=‑3.0e‑12, ES95 errors tighten by ≈0.88 bps, and the sign-test p‑values above confirm the flip-set improvement.

## 2025-11-12T10:10Z — prewhiten RC-lite + coverage lift (feat/prewhiten-coverage@baa1a4b)
- **Data**: `data/returns_daily.csv` (`sha256=96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) + `data/factors/ff5mom_daily.csv` (`sha256=469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), revalidated via `tools/verify_dataset.py` before every AWS dispatch.
- **Commands**:
  1. `make test-fast`
  2. `make aws:rc-lite EXEC_MODE=deterministic`
  3. `INSTANCE_DNS=… KEY_PATH=… RC_REQUIRE_ISOLATED=0 RC_DOW_MIN_REPS=10 make aws:rc-dow`
  4. `INSTANCE_DNS=… KEY_PATH=… RC_VOL_MIN_REPS∈{8,6,4,2} RC_GATE_DELTA_FRAC_MIN∈{0.01,0.007,0.005} RC_OVERLAY_DELTA∈{0.05,0.04,0.03} make aws:rc-vol`
  5. `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251112`
  6. `make memo` and `PYTHONPATH=src:. python tools/build_brief.py --config experiments/equity_panel/config.rc.yaml`
- **Artifacts**:
  - `reports/rc-20251112/{dow-tyler,vol-tyler}/` (full/calm/crisis CSVs, diagnostics detail, `prewhiten_diagnostics.csv`, plots, `resolved_config.json`).
  - `reports/rc-20251112/run_manifest.json` (datasets + AWS run IDs `20251112T035651Z` dow, `20251112T084450Z` vol) and `metrics_summary.json`.
  - Refreshed `reports/memo.md`, `reports/memo_20251112_092343.md`, `reports/brief.md`, `reports/brief_20251112_092348.md`, plus `figures/rc/**`.
- **Coverage / deltas**:
  - DoW (tyler edge, soft gate, q≤2) now lands at detection_rate full ≈ 3.73 %, calm ≈ 4.13 %, crisis ≈ 3.83 % with ΔMSE (overlay vs baseline) ≲3e‑10 and DM stats still undefined because almost every window flips.
  - Vol-state (tyler edge, soft gate, q≤2) is trending upward but still shy of the 2–6 % target: detection_rate full ≈ 0.43 %, calm ≈ 0.14 %, crisis ≈ 0.70 %; `percent_changed` ~10.7 %. Relaxing `group_min_replicates`/`min_reps_vol` down to 2 and lowering `gate_delta_frac_min` to 0.5 % improved acceptance, but the memo + README track the remaining gap as an open item.
- **Notes**:
  - Prewhitening CLI/telemetry is fully wired (CLI flags, `prewhiten_summary.json`, `prewhiten_diagnostics.csv` per window) and covered by `tests/test_equity_prewhiten.py`.
  - Makefile exposes `RC_DOW_MIN_REPS`, `RC_VOL_MIN_REPS`, `RC_VOL_GROUP_REPS`, and `RC_REQUIRE_ISOLATED` so RC-lite targets stay configurable without editing YAML.
  - README Current Status references `reports/rc-20251112/` and punts AWS host specifics to `docs/CLOUD.md` per ops request.
  - Calibration defaults remain untouched; any future tuning still needs the “before/after” note per AGENTS.md.

## 2025-11-05T21:52Z — sprint-1 calibration & smoke (2d235955)
- **Data**: `data/returns_daily.csv` (`sha1=1ff062eab6f0741f7fdc8d25098ffb8f9e3a5344`)
- **Commands**: `make sweep:acceptance`, `make run:equity_smoke`, `make memo`, `make test`
- **Artifacts**: `reports/figures/roc_null.png`, `reports/figures/roc_power.png`, `calibration_defaults.json`, `reports/synthetic/{null_harness,power_harness}/`, `experiments/equity_panel/outputs_smoke/`, `reports/memo.md`
- **Notes**: Synthetic harness writes score tables + calibration defaults (energy floor), detection summary now logs edge bands/gating mode and MV solver stats, MV defaults locked to ridge=1e-4 with box [0,0.1] and 5 bps turnover cost cap.

## 2025-11-07T01:30Z — deterministic DoW + vol RC on WRDS (a03a3764)
- **Data**: `data/returns_daily.csv` (sha256=`96ac7dd3…3197`, verified against `data/registry.json` before every run via `tools/verify_dataset.py`).
- **Commands**: `make test-fast`, `make aws:test-fast MODE=deterministic`, `make aws:rc-dow AWS_ARGS="EDGE=tyler MODE=deterministic"`, `make aws:rc-dow AWS_ARGS="EDGE=scm MODE=deterministic"`, `make aws:rc-vol AWS_ARGS="EDGE=tyler MODE=deterministic"`. (Each rc target uses deterministic BLAS/thread caps via `scripts/aws_run.sh`.)
- **Artifacts**: `reports/rc-20251107/{dow-tyler,dow-scm,vol-tyler}/`, histograms (`acceptance_hist_*.png`, `edge_margin_hist_*.png`), extended DM tables (`dm.csv` now includes LW/OAS contrasts), `reports/rc-20251107/summary_stats.json`, `reports/rc-20251107/run_manifest.json`, calibration log (`reports/calibration_notes.md`), updated `reports/memo.md`.
- **Notes**: All regimes still report 0% acceptance because `detect_spikes` throws `detection_error`; we dropped δ by 0.05 in `calibration/defaults.json` per the crisis decision rule and reran, but failures persisted (documented in `reports/calibration_notes.md`). MV defaults (ridge 1e-4, box [0,0.1], 5 bps turnover, κ-cap 1e6) are enforced everywhere and windows breaching the condition cap are skipped. QLIKE DM tests against LW/OAS remain finite (e.g., DoW full: EW p≈0.003, MV p≈0.009), so the memo now lists those contrasts even though MSE DM stats are undefined. Representative AWS run IDs: `20251107T012713Z` (DoW–Tyler, 45 s), `20251107T012811Z` (DoW–SCM, 45 s), `20251107T012910Z` (Vol–Tyler, 40 s); all completed with status 0 and synced back to `reports/aws/`.

## 2025-11-07T19:25Z — deterministic RC rerun w/ FF5+MOM registry (b1611887)
- **Data**: `data/returns_daily.csv` (sha256 `96ac7dd3…3197`) + `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), both checked via `tools/verify_dataset.py` / `data/factors/registry.json`.
- **Commands**: `make test-fast`; `MODE=deterministic make aws:test-fast` *(blocked: missing `INSTANCE_DNS`)*; `USE_FACTORS=1 EDGE=tyler make rc-dow`; `USE_FACTORS=1 EDGE=scm make rc-dow`; `USE_FACTORS=1 EDGE=tyler make rc-vol`; `EDGE=tyler MODE=deterministic USE_FACTORS=1 make aws:rc-dow` *(blocked: missing `INSTANCE_DNS`)*; `EDGE=tyler MODE=deterministic USE_FACTORS=1 make aws:rc-vol` *(blocked: missing `INSTANCE_DNS`)*.
- **Artifacts**: Refreshed `reports/rc-20251107/{dow-tyler,dow-scm,vol-tyler}/` (each with `run.json`, `resolved_config.json`, updated `diagnostics*.csv`, `dm.csv`, and the new `acceptance_hist_{dow,vol}.png` / `edge_margin_hist_{dow,vol}.png`). Histogram + percent-changed diagnostics feed into the new “Transfer Check” memo section.
- **Notes**: Factor prewhitening now lands on `ff5mom` with `prewhiten_r2_mean ≈ 0.39` and `factor_present_share = 1`. Despite that, overlay gating never fired (`percent_changed = 0` across all regimes; DM `n_effective = 0` and ΔMSE = 0), and reason codes remain dominated by `detection_error` (DoW) and `balance_failure` (vol). Baseline κ̄ stayed mild for DoW (≈11) but vol crisis κ̄ ≈79, highlighting how unbalanced the vol grouping still is. AWS reruns remain blocked until the required SSH environment variables are populated; once available we should re-dispatch `aws:test-fast`, `aws:rc-dow`, and `aws:rc-vol` to capture the same factor-registry outputs on the EC2 runner.

## 2025-11-09T09:10Z — AWS detector telemetry + sensitivity sweep (245fa52)
- **Data**: `data/returns_daily.csv` (sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) and `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), both revalidated via `tools/verify_dataset.py` / `data/registry.json` and the mirrored alias we added to `data/factors/registry.json`.
- **Commands**: `make test-fast`; `INSTANCE_DNS=ec2-98-92-104-129.compute-1.amazonaws.com KEY_PATH=~/.ssh/mateo-us-east-1-ec2-2025 AWS_ARGS="MODE=deterministic" make aws:test-fast`; local RCs for `USE_FACTORS=1 EDGE∈{tyler,scm}` across `rc-dow`, `rc-vol`, `rc-week`, `rc-dowxvol`; deterministic AWS mirrors via `make aws:rc-dow` (tyler + scm), `make aws:rc-vol`, `make aws:rc-week`, `make aws:rc-dowxvol` with `USE_FACTORS=1 EDGE=tyler MODE=deterministic`; `RC_SENS_START=2024-03-01 RC_SENS_END=2024-10-31 make rc-sensitivity`; `make inject-spike` (now routed through `RC_PY`).
- **Artifacts**: 
  - AWS run manifests + telemetry: `reports/aws/20251109T015457Z/runs/20251109T015457Z/run.json` (test-fast), `reports/aws/20251109T015712Z/runs/20251109T015712Z/run.json`, `.../015833Z/...`, `.../020327Z/...`, `.../085041Z/...`, `.../085125Z/...` for the five RC targets.
  - AWS RC outputs: `reports/aws/20251109T015712Z/rc-20251109/dow-tyler/`, `reports/aws/20251109T015833Z/rc-20251109/dow-scm/`, `reports/aws/20251109T020327Z/rc-20251109/vol-tyler/`, `reports/aws/20251109T085041Z/rc-20251109/week/`, `reports/aws/20251109T085125Z/rc-20251109/dowxvol/` (each with `run.json`, `diagnostics.csv`, `diagnostics_detail.csv`, per-regime histograms such as `design_ok_full_hist.png`).
  - Sensitivity sweep: `reports/rc-sensitivity/rc-sensitivity-20251108/{run_manifest.json,tables/sensitivity_summary.csv,tables/changed_windows.csv,figures/acceptance_rate_ri[0|1]_align[0p70|0p80|0p90].png}`.
  - Weak-spike study: `reports/figures/inject_summary.csv`, `reports/figures/{inject_recall.png,inject_fp.png,inject_manifest.json}`.
- **Notes**: All deterministic RCs (including the new week/dow×vol designs) remain at zero acceptance: the enriched telemetry shows `gating_initial = raw_detection_count = 0` everywhere, DoW design compliance tops out at ~59%, vol-state at ~11%, and week slices at ~32%, so reason codes split between `detection_error` and `balance_failure` (see `.../diagnostics_detail.csv`). The sensitivity sweep over 72 `(require_isolated, alignment_min_cos, delta_frac, stability_eta)` combos reports `n_changed_windows = 0` for every cell, so η/δ/alignment tweaks alone cannot unlock detections while the balance issues persist. The weak-spike harness injected μ∈{3,4,5} into ~8% of 1,446 windows yet still logged 0% recall and 0% FP, underscoring that the detector is never emitting candidates on the current balanced panels.

## 2025-11-10T02:08Z — Plan for prewhiten/coverage refresh (284034b1)
- **Scope**: Wire prewhitening + factor diagnostics, relax nested guardrails, rerun RC-lite + ROC sweeps per operator brief.
- **Plan**:
  1. Inspect runners/configs for current prewhiten + diagnostics plumbing; outline manifest schema changes.
  2. Add `--prewhiten/--factor-csv` to eval + panel runners, persist R² + factor names into diagnostics + `run_manifest.json`, update README/memo templates, and cover with unit/integration tests.
  3. Loosen nested balancing constraints, surface `--gate-delta-frac-min`/`q_max` configs, and validate via nested smoke run to ensure skip reasons diversify.
  4. Dispatch deterministic `aws:rc-lite` against WRDS registry data, sync artifacts + telemetry back, and log in PROGRESS.
  5. Execute acceptance ROC sweep (`aws:calibrate-thresholds`), consolidate artifacts + update `calibration_defaults.json`, refresh configs, and prep PR.

## 2025-11-10T03:35Z — Prewhiten plumbing + RC-lite/AWS calibration status (284034b1)
- **Data**: `data/returns_daily.csv` (sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) and `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), verified via `tools/verify_dataset.py` before each local/AWS invocation.
- **Commands**: `pytest tests/experiments/test_prewhiten_utils.py`; `pytest tests/test_equity_prewhiten.py -m slow`; `make test-fast`; `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --prewhiten ff5mom --factor-csv data/factors/ff5mom_daily.csv --no-progress --exec-mode deterministic` (local nested smoke to validate relaxed balancing + per-window factor telemetry); `INSTANCE_DNS=ec2-98-92-104-129.compute-1.amazonaws.com KEY_PATH=~/.ssh/mateo-us-east-1-ec2-2025 AWS_ARGS="EXEC_MODE=deterministic" make aws:rc-lite`; `INSTANCE_DNS=... AWS_ARGS="EXEC_MODE=deterministic CALIB_TRIALS_NULL=600 CALIB_TRIALS_ALT=600" make aws:calibrate-thresholds` (in flight; monitor shows `calibration_progress: 1/48` at run id `20251110T031745Z`).
- **Artifacts**:
  - Local nested smoke: `experiments/equity_panel/outputs_nested_smoke/nested_J5_solver-auto_est-dealias_prep-prewhiten_factorsMKT,SMB,HML,RMW,CMA,MOM-prewhiten_modeff5mom/{rolling_results.csv,summary.json,run_meta.json}` now carry `prewhiten_*` columns plus relaxed nested skip detail (`weeks_common`, `years_dropped`, `replicates_used`).
  - RC-lite AWS batch (run dir `reports/aws/20251110T025119Z/`): refreshed smoke/crisis outputs with memo+brief regenerated (`reports/aws/20251110T025119Z/memo_20251110_025534.md`, `.../brief.md`) and gallery snapshots (`figures/rc/**`). Each run folder contains the augmented diagnostics columns (`prewhiten_*`, `factor_present`).
  - AWS calibration sweep is still running under `reports/aws/runs/20251110T031745Z` (not yet synced locally); progress + logs available via `reports/runs/monitor` once the EC2 job finishes. No new `calibration/edge_delta_thresholds.json` committed yet—pending that run’s completion.
- **Notes**: Evaluation + panel runners accept `--prewhiten`/`--factor-csv` and persist telemetry into diagnostics, summary payloads, and `run.json`. Memo/brief templates now expose a “Factor Baseline” block. Nested guardrails accept looser replicate/ISO-week intersections (records `replicates_used`, `years_dropped`) and daily runners surface `RC_Q_MAX`/`RC_GATE_DELTA_FRAC_MIN`. `make test-fast` + the new targeted tests cover the refactor. RC-lite on AWS completed deterministically; calibration sweep remains queued (ETA ≈ 9h per `run_monitor`), so defaults will be updated once that run lands.

## 2025-11-11T01:45Z — Deterministic AWS calibration sweep (20251110T154048Z)
- **Data**: Same WRDS returns + FF5+MOM factors as prior RC runs (hashes above), re-verified before dispatch.
- **Commands**: `INSTANCE_DNS=ec2-98-92-104-129.compute-1.amazonaws.com KEY_PATH=~/.ssh/mateo-us-east-1-ec2-2025 AWS_ARGS="EXEC_MODE=deterministic CALIB_TRIALS_NULL=600 CALIB_TRIALS_ALT=600" make aws:calibrate-thresholds`; post-run rsync of `reports/runs/20251110T154048Z/` and `calibration/*`; `make test-fast`.
- **Artifacts**:
  - Remote provenance: `reports/aws/20251110T154048Z/runs/20251110T154048Z/run.json` (status `0`, duration ≈10 h) plus full `metrics.jsonl`/`progress.jsonl`.
  - Updated calibration files in repo: `calibration/edge_delta_thresholds.json` (48 cells) + `calibration/defaults.json` (new selection + metadata).
  - Local monitor log `aws_calib_latest.log` capturing `[1/48 … 48/48]` milestones.
- **Notes**: Sweep covered `(p ∈ {64, 80, 96}, replicates ∈ {14, 20}, δ_abs ∈ {0.35, 0.45, 0.55, 0.65}, edge ∈ {scm, tyler})` under deterministic thread caps. Final ETA ticked to zero at 48/48 with no retries. With new thresholds committed, RC configs should continue pointing at `calibration/edge_delta_thresholds.json` (rev 20251110). Unit tests re-run locally (`make test-fast`) to confirm no regressions after syncing the calibration outputs.

## 2025-11-11T02:00Z — RC-lite refresh w/ new calibration defaults (284034b1)
- **Data**: `data/returns_daily.csv` + `data/factors/ff5mom_daily.csv` (hashes above), verified pre-run.
- **Commands**: `make rc-lite` (covers smoke + 2020 crisis configs across {dealias,lw,oas}); gallery/memo/brief regenerated implicitly by the target.
- **Artifacts**:
  - Smoke outputs: `experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-{dealias,lw,oas}_prep-*`.
  - Crisis outputs: `experiments/equity_panel/outputs_crisis_2020/...`.
  - Gallery: `figures/rc/**` (new tables/plots incorporate `prewhiten_*` columns).
  - Memo/brief refreshed under `reports/{memo.md,brief.md}` plus timestamped copies.
- **Notes**: First full rc-lite after ingesting the 20251110 calibration defaults. No runtime errors; diagnostics now show factor baselines + updated Δ thresholds. Ready to mirror on AWS (`make aws:rc-lite`) if we want cloud telemetry.

## 2025-11-21T08:15Z — Nested guardrails + RC-lite refresh (a94ad00)
- **Data**: WRDS `data/returns_daily.csv` + `data/factors/ff5mom_daily.csv` (registry hashes unchanged).
- **Commands**:
  - Nested smoke (throughput, tuned guard): `.venv/bin/python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --no-progress --workers 32 --assets-top 80 --stride-windows 4 --cache-dir .cache --precompute-panel --drop-partial-weeks --estimator dealias --prewhiten ff5mom --use-factor-prewhiten 1` (detection_rate=1/24).
  - RC-lite (WRDS): `PATH=.venv/bin:$PATH RC_WORKERS=$(nproc) EXEC_MODE=throughput make rc-lite` followed by `make memo` to rebuild gallery/memos (latest: `reports/memo_20251121_081543.md`).
  - Acceptance sweep: `PATH=.venv/bin:$PATH EXEC_MODE=throughput HARNESS_TRIALS=400 make sweep:acceptance` (calibration_defaults.json regenerated; ROC figs under `reports/figures/`).
  - Ablations: `python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml` (assets_top=60, 2020–2021 slice) and `.venv/bin/python experiments/equity_panel/run.py --config experiments/equity_panel/config.ablation.smoke.yaml --no-progress --workers 8 --assets-top 60 --stride-windows 4 --cache-dir .cache --precompute-panel --drop-partial-weeks --estimator dealias --prewhiten ff5mom --use-factor-prewhiten 1 --ablations`.
  - Tests: `PATH=.venv/bin:$PATH make test-fast` (65 passed).
- **Results**:
  - Nested coverage now in-band: `experiments/equity_panel/outputs_nested_smoke/.../summary.json` shows detection_windows=1/24 (4.17%) with `nested_guard` skips=5 after applying stability/edge floor of 3 bps.
  - RC-lite artifacts refreshed in place (`figures/rc`, memo at `reports/memo_20251121_081543.md`); nested plots reflect the new guardrails.
  - Synthetic acceptance defaults re-written (timestamp 2025-11-21T03:30Z) via sweep; thresholds unchanged numerically (delta_frac=0.02, eta=0.4, energy_floor≈0.1012).
  - Ablation assets/regime aligned to RC (2020–2021 calm+crisis): updated grid in `experiments/ablate/ablation_matrix_tiny.yaml`, matrix at `ablations/ablation_matrix.csv`, and E5 summary at `experiments/equity_panel/outputs_ablation_smoke/oneway_J5.../ablation_summary.csv`.

## 2025-11-20T23:43Z — Nested gating tweak + ablation plumbing (68c1c6d)
- **Data**: WRDS daily returns `data/returns_daily.csv` and factors `data/factors/ff5mom_daily.csv` (hashes unchanged).
- **Commands**: 
  - `.venv` bootstrap (`python3 -m venv .venv && pip install -e .[dev]`), `make test-fast` (65 passed).
  - Nested smoke reruns: `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --no-progress --exec-mode deterministic --factor-csv data/factors/ff5mom_daily.csv --prewhiten ff5mom --use-factor-prewhiten 1 --estimator dealias` (multiple passes after gating relax/threshold tweaks).
  - Ablation attempts: `make rc-ablations` and `python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml` (timed out on this host after ≥10 min per attempt; no new ablation artifacts written yet).
- **Results/Notes**: Nested gating now records non-isolated fallback telemetry and is enabled via config, but the current nested smoke still reports 0/24 detection windows (no candidates emitted by `dealias_search`). Ablation grid defaults switched to the tiny matrix and hooked into `rc-ablations`, but runs remain long locally; expect faster completion on Hetzner once queued. No changes to calibration files. Next steps: run ablate tiny matrix on Hetzner or further shrink window/asset subset; investigate why nested detection remains zero despite relaxed stability/delta and non-isolated fallback.

## 2025-11-21T02:10Z — Hetzner ablations + nested coverage unlocked (d1b39ed)
- **Data**: Same WRDS returns/factors as prior entries.
- **Commands**:
  - Nested smoke (throughput): `PYTHONPATH=src OMP_NUM_THREADS=4 EXEC_MODE=throughput python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --no-progress --exec-mode throughput --factor-csv data/factors/ff5mom_daily.csv --prewhiten ff5mom --use-factor-prewhiten 1 --estimator dealias`.
  - Ablation matrix (tiny grid): `EXEC_MODE=throughput python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml`.
  - Fast E5 ablation drop (direct call to _run_param_ablation): inline Python snippet loading `data/returns_daily.csv` (2023-01-03→2023-02-10) and writing `experiments/equity_panel/outputs_ablation_smoke/ablation_summary.csv`.
  - Gallery/memo refresh: `python tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml`; `python tools/build_memo.py --config experiments/equity_panel/config.rc.yaml`.
  - Tests: `.venv` active; `make test-fast` (65 passed).
- **Results**:
  - Nested smoke now logs 7/24 detection windows (29% coverage) with gating skips recorded; edge margin median ~0.023; non-isolated fallback not triggered; summary at `experiments/equity_panel/outputs_nested_smoke/.../summary.json`.
  - Ablation artifacts: `ablations/ablation_matrix.csv` (4-combo tiny grid) and `experiments/equity_panel/outputs_ablation_smoke/ablation_summary.csv`; gallery/memo pick up the matrix (heatmap/table) and ablation summary directory.
  - **Notes**: Added `use_tvector` toggle (configurable, disabled for nested smoke and ablations) to bypass overly strict t-vector gating; relaxed nested thresholds (delta_frac 0.005, eps 0.01, eta 0.15, q_max 2, require_isolated=false). Equity-panel ablation runner still heavy when invoked via `rc-ablations`; direct `_run_param_ablation` was used to emit the E5 summary for this drop. Next: rerun the full `rc-ablations` target on Hetzner if time permits, or wire timeout/limit guards.

## 2025-12-19T23:27Z — Weekly guardrail attribution (ticket-09)
- **Data**: WRDS daily returns (`experiments/equity_panel/config.smoke.yaml`), deterministic exec mode.
- **Commands**:
  - PATH="/root/fjs-dealias-portfolio/.venv/bin:$PATH" make test-fast
  - PYTHONPATH=src EXEC_MODE=deterministic .venv/bin/python experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml --gating-mode fixed --minvar-ridge 0.0001 --minvar-box 0.0,0.1 --minvar-condition-cap 1000000000 --turnover-cost 5 --gating-diagnostics --output-dir experiments/equity_panel/outputs_ticket-09_20251219_232717
  - PYTHONPATH=src .venv/bin/python tools/summarize_weekly_diagnostics.py --input experiments/equity_panel/outputs_ticket-09_20251219_232717/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/gating_diagnostics.csv
- **Results**:
  - Replaced `guard_other` with explicit diagnostic keys and added stable `skip_reason_primary/detail/exception_type` across gating outputs; diagnostic failures now carry exception type/message.
  - Gating diagnostics test updated to enforce no `guard_other` and require detail for `diagnostic_failure`.
  - Smoke output: detection_rate=75% with one `no_isolated_spike`; guard totals tvec_compute_error=72, tvec_target_zero=2, tvec_off_component=1074. Weekly summary at `experiments/equity_panel/outputs_ticket-09_20251219_232717/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/weekly_diagnostics.md`.

## 2025-12-20T04:59Z — Eval contamination hardening (ticket-11)
- **Commands**: 
  - python3 -m pip install --break-system-packages pytest numpy pandas scipy matplotlib scikit-learn jinja2
  - make test-fast
  - EXEC_MODE=deterministic PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py --returns-csv data/returns_daily.csv --window 40 --horizon 5 --max-windows 4 --group-min-replicates 2 --assets-top 30 --prewhiten off --use-factor-prewhiten 0 --out reports/eval-ticket-11-smoke-small
- **Changes**: Enforced aligned window intersections for all Δ metrics and DM tests, added `n_effective_*` and `comparison_valid` flags, surfaced cap/truncation sources (max_windows, coverage, condition caps, date truncation) in `run.json` windows block, and emitted per-estimator skip-share tables (`skip_stats.csv`). New config knob `min_comparison_windows` (default 30) documented; summary tooling now carries cap/coverage and invalid-comparison warnings.
- **Artifacts**: `reports/eval-ticket-11-smoke-small/` (capped, deterministic): check `run.json` windows block, `full/dm.csv` and `full/metrics.csv` for aligned n_effective/comparison_valid, and `skip_stats.csv` for skip shares.

## 2025-12-20T07:46Z — Eval contamination fixup (ticket-15)
- **Branch / git**: codex/ticket-15-eval-contamination-fixup @ 35242b0
- **Commands**:
  - `. .venv/bin/activate && make test-fast`
  - `EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --out reports/eval-ticket-15-smoke-aligned5 --assets-top 20 --window 42 --horizon 10 --max-windows 50 --min-comparison-windows 30 --prewhiten off --overlay-delta 0.0 --gate-mode soft --gate-accept-nonisolated`
- **Outputs**: `reports/eval-ticket-15-smoke-aligned5/` (capped: max_windows=50, window_coverage≈0.013; comparison_valid_mse/es/qlike=1, DM n_effective=0). Not headline—capped/truncated run excluded from aggregates by completeness rules.
- **Notes**: Added per-metric `comparison_valid_*` flags and `windows_after_caps` to eval metadata; DM comparison_valid now respects `min_comparison_windows`; summary sanity aggregation now explicitly drops capped runs.

## 2025-12-20T20:36Z — Weekly gating diagnostics attribution (ticket-09)
- **Branch / git**: codex/ticket-09-gating-diagnostics-attribution @ 39e2889
- **Commands**: make test-fast; PYTHONPATH=/root/fjs-dealias-portfolio:/root/fjs-dealias-portfolio/src RUN_NAME=$RUN_NAME python3 docs/agent_runs/$RUN_NAME/synth_diag_failure.py; PYTHONPATH=/root/fjs-dealias-portfolio:/root/fjs-dealias-portfolio/src python3 tools/summarize_weekly_diagnostics.py --input experiments/equity_panel/outputs_ticket-09_synth_failure_$RUN_NAME/gating_diagnostics.csv; EXEC_MODE=deterministic make run:equity_smoke; PYTHONPATH=/root/fjs-dealias-portfolio:/root/fjs-dealias-portfolio/src python3 tools/summarize_weekly_diagnostics.py --input experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/gating_diagnostics.csv
- **Outputs**: synthetic diagnostic-failure run `experiments/equity_panel/outputs_ticket-09_synth_failure_$RUN_NAME/`; real smoke `experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/`; run log `docs/agent_runs/$RUN_NAME/`.
- **Notes**: Added structured exception_stage/message_short fields and replicates to gating_diagnostics, tightened diagnostic_failure attribution (exception_type + detail, no guard_other), and expanded weekly_diagnostics to show reason counts plus top-5 window examples with fit/hold ranges and guard stats. `run:equity_smoke` now uses python3 and emits gating diagnostics by default.
## 2025-12-21T20:08Z — ticket-04 paper-v1 ablation runner (uncapped SCM/OAS/RIE × overlay on/off)
- **Branch/Run**: `feat/ticket-04-paper-v1-ablation` (RUN_NAME=`20251221_194517_ticket-04_paper-v1-ablation`), git sha `7f126a9b143be5545cc5b9d0bdf99d09777b1066`.
- **Commands**: `make test-fast`; `EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate`.
- **Changes**: Added pinned daily config `experiments/eval/config.paper_v1.yaml` (DoW, min_comparison_windows=50, MV constraints, overlay grid=30); added `rc-paper-v1-ablate` Make target; implemented `tools/paper_v1_ablation.py` + unit test; summary tables now include delta_qlike + comparison_valid_{mse,es,qlike} and per-run rows when rc dir contains subruns; eval runner short-circuits detections when `q_max<=0` for explicit overlay OFF runs.
- **Artifacts**: RC root `reports/rc-paper-v1-ablate-20251221_205751/`; combined table `reports/rc-paper-v1-ablate-20251221_205751/summary/paper_v1_ablation.csv`; run log `docs/agent_runs/20251221_194517_ticket-04_paper-v1-ablation/`.
- **What we learned**:
  - Overlay OFF runs (q_max=0) produce zero detection_rate and preserve cap_active=false across all six legs.
  - With the 196‑date real-data subset, some comparisons fall below min_comparison_windows; `limitations.md` flags insufficient aligned windows (not capped).
  - Overlay ON runs show ~3.3% detection_rate on the subset, with consistent accepted gating across shrinkers.
## 2025-12-22T07:54Z — ticket-06 window_coverage planning fix (daily DoW paper v1)
- **Branch/Run**: `feat/ticket-06-window-coverage` (RUN_NAME=`20251222_014730_ticket-06_window-coverage`), git sha `8a5579b8f34176b43c75543c0a3305f8a8fe2aa2`.
- **Commands**: `make test-fast`; `PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-06-20251222_063304/dow-paper-v1 --exec-mode deterministic`; `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-06-20251222_063304`.
- **Results**: window_coverage no longer flags uncapped runs when the only missing windows are holdout-empty; run.json now tracks candidate/planned windows and holdout drops, and limitations.md surfaces the dropped count. The paper-v1 daily DoW run is headline-eligible (`cap_active=false`, `window_coverage=1.0`, summary tables non-empty, comparison_valid=1, n_effective>=715). Pre-fix evidence: ticket-05 run flagged `cap_active=true` with `cap_sources=['window_coverage']` due to 115 holdout-empty windows lacking identifiers; post-fix run logs `windows_dropped_holdout_empty=115` without capping.
- **Artifacts**: run log `docs/agent_runs/20251222_014730_ticket-06_window-coverage/`; outputs `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`; summary `reports/rc-ticket-06-20251222_063304/summary/`.

## 2025-12-22T19:56Z — ticket-07 advisor-ready daily DoW paper-v1 rerun
- **Branch/Run**: `feat/ticket-07-advisor-ready-dow` (RUN_NAME=`20251222_183526_ticket-07_advisor-ready-dow`), git sha `2cb5bfdce66324fff011d994d552a4b9bc42740c`.
- **Commands**: `make test-fast`; `PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-07-20251222_183800/dow-paper-v1 --exec-mode deterministic`; `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-07-20251222_183800`; `python3 scripts/check_data_policy.py`; `rg -n "WRDS_PASSWORD|WRDS_USERNAME|password|token|secret" -S .`; `git ls-files | xargs rg -n "strike,.*market_iv|\\bsecid\\b|best_bid|best_ask|best_offer" -S`.
- **Results**: headline-eligible uncapped run with `cap_active=false`, `window_coverage=1.0`, `windows_requested=3512`, `windows_evaluated=3512`, and `windows_dropped_holdout_empty=115` (logged as `holdout_empty`). Full-regime detection_rate_mean=0.04162 (1751/1774) and percent_changed=100%; n_effective=1749 with comparison_valid_* = 1 for EW/MV. Full-regime deltas: EW ΔQLIKE=-0.06719 (ΔMSE=+2.64e-11), MV ΔQLIKE=-0.03576 (ΔMSE=-6.65e-13). Limitations note holdout-empty drops but no caps.
- **Artifacts**: run log `docs/agent_runs/20251222_183526_ticket-07_advisor-ready-dow/`; outputs `reports/rc-ticket-07-20251222_183800/dow-paper-v1/`; summary `reports/rc-ticket-07-20251222_183800/summary/` (includes `advisor_snapshot.md`).

## 2025-12-23T06:02Z — ticket-16 paper config integrity
- **Branch/Run**: `codex/ticket-16-paper-config-integrity` (RUN_NAME=`20251223_064432_ticket-16_paper-config-integrity`), git sha `361c869c8f35ab34d43f7346a4ad3afccf1fdc3a`.
- **Commands**: `make test-fast`; `EXEC_MODE=deterministic make rc-lite-sanity`.
- **Changes**: explicit eval config paths now fail loudly when missing; `run.json` records `resolved_config_path`, `resolved_config_hash`, and `git_dirty`; added tests for missing/paper configs; removed the missing paper-v1 config Known Issue and marked the PLAN_OF_RECORD roadmap item as done.
- **Artifacts**: run log `docs/agent_runs/20251223_064432_ticket-16_paper-config-integrity/`; outputs `reports/rc-20251223-sanity-20251223_064808/`; weekly outputs `experiments/equity_panel/outputs_rc-lite-20251223_20251223_064808/`.

## 2025-12-25T23:38Z — ticket-23 inject-spike diagnostics + max-windows
- **Branch/Run**: `codex/ticket-23-inject-spike-diagnostics-maxwindows` (RUN_NAME=`20251225_224205_ticket-23_inject-spike-diagnostics-maxwindows`).
- **Commands**:
  - `python -m pytest tests/experiments/test_inject_spike.py -q`
  - `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --config experiments/eval/config.yaml --group-design dow --assets-top 80 --start 2022-01-01 --end 2022-12-31 --mu-grid 3,6,12,24 --max-windows 25 --window-sampling random --seed 23 --run-id 20251225_ticket23_dow_tyler`
  - Multiple week/scm/coarse attempts were started and aborted due to long `dealias_search` runtime (see run log COMMANDS for exact invocations).
- **Results/Notes**:
  - `inject_spike` now writes `windows_detail.csv` + `gating_reasons.csv` with guard counts; `run.json` includes sampling metadata + reason-bucket summaries.
  - DoW run (20251225_ticket23_dow_tyler) remains flat-zero across μ; gating reasons dominated by `tvec_compute_error` + `tvec_off_component` (pre-gate), indicating the t-vector guardrail blocks candidates.
  - Week smokes did not finish locally; no completed week curve yet (aborted runs left only `resolved_config.json`).
- **Artifacts**:
  - `reports/inject_spike/20251225_ticket23_dow_tyler/` (curve/plot/run.json/windows_detail/gating_reasons)
  - `docs/agent_runs/20251225_224205_ticket-23_inject-spike-diagnostics-maxwindows/artifacts/curve_dow_tyler.csv`
  - `docs/agent_runs/20251225_224205_ticket-23_inject-spike-diagnostics-maxwindows/artifacts/gating_reasons_dow_tyler.csv`
- **Key results (DoW)**:
  - μ=0.0: detect=0.00, accept=0.00 (n_windows=25)
  - μ=3.0: detect=0.00, accept=0.00 (n_windows=1)
  - μ=6.0: detect=0.00, accept=0.00 (n_windows=1)
  - μ=12.0: detect=0.00, accept=0.00 (n_windows=1)
  - μ=24.0: detect=0.00, accept=0.00 (n_windows=1)
- **Key results (Week)**:
  - No completed run (local runtime aborts during `dealias_search`).

## 2025-12-26T08:15Z — ticket-24 finish week inject-spike diagnostics
- **Branch/Run**: `codex/ticket-24_finish-week-inject-spike` (RUN_NAME=`20251226_060917_ticket-24_finish-week-inject-spike`, run_id=`20251226_ticket24_week_full_fix`), git sha `31c05a57ffd5db7a1531c427eb7373de5f7a5f22`.
- **Commands**:
  - `EXEC_MODE=throughput OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --config docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/inject_spike_fast.yaml --group-design week --assets-top 20 --window 30 --horizon 5 --mu-grid 3,6,12,24 --inject-frac-min 0.4 --inject-frac-max 0.4 --profile --run-id 20251226_ticket24_week_full_fix`
  - `python -m pytest tests/experiments/test_inject_spike.py -q`
  - `make test-fast`
- **Results**:
  - Week full run remains flat-zero across μ (curve in run log artifacts). Dominant pre-gate reasons: `tvec_off_component` (22320), `tvec_no_real_root` (7756), `tvec_no_admissible_root` (3404); `tvec_compute_error=0` after the classification fix.
  - Runtime: ~879s wall (throughput, 1 worker; BLAS threads pinned to 1). Profile shows `mp.t_vec`/`admissible_m_from_lambda` dominating.
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/`
  - Outputs `reports/inject_spike/20251226_ticket24_week_full_fix/`
  - Review copies `docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/`

## 2025-12-26T08:15Z — ticket-24 finish week inject-spike diagnostics
- **Branch/Run**: `codex/ticket-24_finish-week-inject-spike` (RUN_NAME=`20251226_060917_ticket-24_finish-week-inject-spike`, run_id=`20251226_ticket24_week_full_fix`), git sha `31c05a57ffd5db7a1531c427eb7373de5f7a5f22`.
- **Commands**:
  - `EXEC_MODE=throughput OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --config docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/inject_spike_fast.yaml --group-design week --assets-top 20 --window 30 --horizon 5 --mu-grid 3,6,12,24 --inject-frac-min 0.4 --inject-frac-max 0.4 --profile --run-id 20251226_ticket24_week_full_fix`
  - `python -m pytest tests/experiments/test_inject_spike.py -q`
  - `make test-fast`
- **Results**:
  - Week full run remains flat-zero across μ (curve in run log artifacts). Dominant pre-gate reasons: `tvec_off_component` (22320), `tvec_no_real_root` (7756), `tvec_no_admissible_root` (3404); `tvec_compute_error=0` after the classification fix.
  - Runtime: ~879s wall (throughput, 1 worker; BLAS threads pinned to 1). Profile shows `mp.t_vec`/`admissible_m_from_lambda` dominating.
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/`
  - Outputs `reports/inject_spike/20251226_ticket24_week_full_fix/`
  - Review copies `docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/artifacts/`

## 2025-12-26T09:03Z — ticket-25 component-aware inject-mode + week between smoke
- **Branch/Run**: `codex/ticket-25_inject-component-modes` (RUN_NAME=`20251226_095630_ticket-25_week-between-smoke`), git sha `3c347a1`.
- **Commands**: `python -m pytest tests/experiments/test_inject_spike.py -q`; `make test-fast`; `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,12,24 --inject-mode between --max-windows 20 --window-sampling random --window-sampling-seed 7 --seed 7 --run-id 20251226_095630_ticket-25_week-between-smoke --out reports/inject_spike`.
- **Results**: Added inject_mode {total,between,within} with group-aware injection series + metadata/CSV updates. Week between-mode smoke shows detect/accept=1.00 at μ=12 and μ=24 (2/2 injected windows), baseline detect/accept=0; pre-gate reasons dominated by tvec_off_component/tvec_no_real_root/tvec_no_admissible_root at μ=0.
- **Artifacts**: run log `docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/`; outputs `reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/`.

## 2025-12-26T10:33Z — ticket-25 within/total fixture smokes
- **Branch/Run**: `codex/ticket-25_inject-component-modes` (RUN_NAME=`20251226_102602_ticket-25_week-within-total-smoke`), git sha `e198fddf2cb74df8e41b5a9d043e45f300115aae`.
- **Commands**: `python -m pytest tests/experiments/test_inject_spike.py -q`; `make test-fast`; `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,12,24 --inject-mode within --max-windows 20 --window-sampling random --window-sampling-seed 7 --seed 7 --run-id 20251226_102602_ticket-25_week-within-smoke --out reports/inject_spike`; `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,12,24 --inject-mode total --max-windows 20 --window-sampling random --window-sampling-seed 7 --seed 7 --run-id 20251226_102602_ticket-25_week-total-smoke --out reports/inject_spike`.
- **Results**: Within and total smokes remain flat-zero at μ=12/24 (detect/accept=0); μ=24 raw_outliers_found share=0. Dominant pre-gate reasons remain tvec_off_component, tvec_no_real_root, tvec_no_admissible_root. Filled missing TESTS.md for the prior between-mode run log.
- **Artifacts**: run log `docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/`; outputs `reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/` and `reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/`.

## 2025-12-26T10:05Z — ticket-25 between stress test (fixture)
- **Branch/Run**: `codex/ticket-25_inject-component-modes` (RUN_NAME=`20251226_105628_ticket-25_week-between-stress`), git sha `d9b0e2d09ce19226c019feba187ae9c7b742d28c`.
- **Commands**: `python -m pytest tests/experiments/test_inject_spike.py -q`; `make test-fast`; `PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,6,12,18,24,30,36 --inject-mode between --inject-frac-min 0.2 --inject-frac-max 0.2 --max-windows 48 --window-sampling first --window-sampling-seed 7 --seed 7 --run-id 20251226_105628_ticket-25_week-between-stress --out reports/inject_spike`.
- **Results**: Using all 48 windows and fixed inject_frac=0.2, between-mode shows detect/accept=1.00 for μ=6–36 (10/10 injected windows each), baseline detect/accept=0. Dominant pre-gate reasons at μ=0 remain tvec_off_component/tvec_no_real_root/tvec_no_admissible_root; μ=36 has 10 accepted post-gate.
- **Artifacts**: run log `docs/agent_runs/20251226_105628_ticket-25_week-between-stress/`; outputs `reports/inject_spike/20251226_105628_ticket-25_week-between-stress/`.

## 2025-12-26T18:36Z — ticket-19 changed-window reporting
- **Branch/Run**: `codex/ticket-19_changed-window-reporting` (RUN_NAME=`20251226_174844_ticket-19_changed-window-reporting`), git sha `5980a47d819624954a422b672f228fee16a8f61b`.
- **Commands**:
  - `make test-fast`
  - `EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity`
  - `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251226-sanity-20251226_191833`
- **Results**:
  - Added changed-window-only ΔMSE/ΔQLIKE, `n_changed`, `changed_frac`, and median weight-delta stats to summaries; limitations now include a conditional-reporting section.
  - Evaluation now always aligns Δ metrics/DM stats on changed-window sets; added per-window weight-delta diagnostics in `metrics_detail.csv`.
  - rc-lite-sanity summaries were generated but excluded from aggregates due to date-truncation caps (summary_perf header-only).
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/`
  - Outputs `reports/rc-20251226-sanity-20251226_191833/`
  - Weekly smoke outputs `experiments/equity_panel/outputs_rc-lite-20251226_20251226_191833/`
  - Bundle `docs/gpt_bundles/20251226_194611_ticket-19_20251226_174844_ticket-19_changed-window-reporting.zip`


## 2025-12-26T20:16Z — ticket-20 uncapped RC eval (changed-window stats)
- **Branch/Run**: `codex/ticket-20_uncapped-rc-week` (RUN_NAME=`20251226_191530_ticket-20_uncapped-rc-week`), git sha `c8b95a67fc8e24b881dedb5b1c9fc9ab8e3ccc63`.
- **Commands**:
  - `make test-fast`
  - `python tools/verify_dataset.py data/returns_sample.csv --registry data/registry.json`
  - `python tools/verify_dataset.py data/returns_sample_spike.csv --registry data/registry.json`
  - `python tools/verify_dataset.py data/factors/ff5mom_daily.csv --registry data/factors/registry.json`
  - `EXEC_MODE=throughput PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py --returns-csv data/returns_sample_spike.csv --window 40 --horizon 10 --assets-top 8 --group-design week --group-min-count 2 --group-min-replicates 2 --edge-mode tyler --shrinker rie --prewhiten ff5mom --overlay-delta 0.05 --coarse-candidate 1 --gate-mode soft --gate-accept-nonisolated --gate-stability-min 0.0001 --allow-non-isolated --use-factor-prewhiten 1 --gate-delta-calibration calibration/edge_delta_thresholds.json --gate-delta-frac-min 0.02 --q-max 2 --mv-gamma 1e-4 --mv-box 0.0,0.1 --mv-turnover-bps 5 --mv-condition-cap 1000000 --workers $(nproc) --out reports/rc-20251226/sample_spike_uncapped`
  - `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251226/sample_spike_uncapped`
- **Results**:
  - Added fixture datasets `data/returns_sample.csv` and `data/returns_sample_spike.csv` to support fast uncapped validation; updated `data/registry.json`.
  - Final uncapped run (sample_spike_uncapped) produces changed-window stats in summary_perf (n_changed > 0; changed_frac=1.0 due to injected spikes).
  - Longer full-dataset uncapped runs were aborted after ~20 min without completing outputs.
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_191530_ticket-20_uncapped-rc-week/`
  - Outputs `reports/rc-20251226/sample_spike_uncapped/` (summary in `summary/summary_perf.csv`)

## 2026-01-10T10:34Z — ticket-20 uncapped RC eval reruns
- **Branch/Run**: `codex/ticket-20_uncapped-rc-week` (RUN_NAME=`20251226_191530_ticket-20_uncapped-rc-week`), git sha `b5e0f4b986b8fb2d25c1de767f97c3655df221f5`.
- **Commands**:
  - `make test-fast`
  - `PYTHONUNBUFFERED=1 EXEC_MODE=deterministic PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py --returns-csv data/returns_daily.csv --window 126 --horizon 21 --assets-top 80 --group-design week --group-min-count 4 --group-min-replicates 1 --edge-mode tyler --shrinker rie --prewhiten ff5mom --overlay-delta 0.05 --coarse-candidate 1 --gate-mode soft --gate-accept-nonisolated --gate-stability-min 0.0001 --require-isolated --use-factor-prewhiten 1 --gate-delta-calibration calibration/edge_delta_thresholds.json --gate-delta-frac-min 0.02 --q-max 2 --mv-gamma 1e-4 --mv-box 0.0,0.1 --mv-turnover-bps 5 --mv-condition-cap 1000000 --workers $(nproc) --out reports/rc-20260110/week_uncapped_full_minrep1_rerun2 2>&1 | tee reports/rc-20260110/week_uncapped_full_minrep1_rerun2/run.log`
- **Results**:
  - Full uncapped reruns on returns_daily exited after prewhiten with no eval outputs (no run.json/metrics; run.log empty), so summary_perf could not be generated.
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_191530_ticket-20_uncapped-rc-week/`
  - Outputs `reports/rc-20251230/week_uncapped_full_minrep1/`, `reports/rc-20251230/week_uncapped_full_minrep1_rerun/`, `reports/rc-20260110/week_uncapped_full_minrep1_rerun2/`
