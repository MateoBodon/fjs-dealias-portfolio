## 2025-11-21T08:15Z
- **Step**: Added nested guardrails (edge/stability floors) to gating, reran nested smoke + RC-lite on WRDS, refreshed memos/gallery, and regenerated synthetic acceptance + ablations.
- **Decisions**: Nested config now uses delta_frac=0.012, eta=0.22, energy_min_abs=5e-7, off_component_leak_cap=25, `require_isolated=false` plus a guard of edge_margin/stability_margin ≥3 bps before acceptance. Ablation-only runs now slice the prewhitened panel to the config start/end window; tiny ablation grid trimmed to assets_top=60 and 2020–2021 slice.
- **Checks**: `PATH=.venv/bin:$PATH make test-fast`; `make rc-lite` + `make memo`; `HARNESS_TRIALS=400 make sweep:acceptance`; `python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml`; ablation smoke rerun for 2020-03→2020-06 (assets_top=60).
- **Highlights**:
  - Nested smoke coverage now 1/24 windows (4.17%) with `nested_guard` skips=5; detection and gating telemetry in `experiments/equity_panel/outputs_nested_smoke/.../summary.json`.
  - RC-lite artifacts refreshed in place (`figures/rc`, `reports/memo_20251121_081543.md`); DoW/crisis runs unchanged, nested plots reflect the new guard.
  - Calibration defaults rebuilt at `2025-11-21T03:30Z` (numerics unchanged: delta_frac=0.02, eta=0.4, energy_floor≈0.1012); ROC figures under `reports/figures/`.
  - Ablations aligned to RC regimes: updated grid (`experiments/ablate/ablation_matrix_tiny.yaml`, assets_top=60, 2020–2021) and equity-panel E5 summary at `experiments/equity_panel/outputs_ablation_smoke/.../ablation_summary.csv`.

## 2025-11-13T04:05Z
- **Step**: Executed the deterministic AWS pair for vol-state (top‑80, 126×21) with the new gating knobs, captured flip-set deltas, and published manifests + docs.
- **Decisions**: `rc-vol` defaults to `--allow-non-isolated` plus the q≥2 alignment guard (`VOL_Q2_ALIGNMENT_MIN_COS=0.9`), so we ran one pass with `USE_FACTORS=0` (`run_id=20251112T232711Z`) and one with `USE_FACTORS=1` (`run_id=20251113T014417Z`) using the exact command strings recorded in `reports/runs/<RUN_ID>/run.json`. Outputs were copied into `reports/rc-20251113/{vol-off,vol-ff5mom}/` and summarised via the new `prewhiten_effect.py`.
- **Checks**: `python tools/verify_dataset.py …` for both returns and factors, `make test-fast`, `make gallery`, `make memo`, plus manual inspection of `reports/runs/20251112T232711Z/metrics_summary.json` and `reports/runs/20251113T014417Z/metrics_summary.json` to ensure telemetry captured steady CPU/RSS and deterministic thread caps.
- **Highlights**:
  - Acceptance sits in-band: `reports/rc-20251113/vol-off/full/diagnostics.csv` shows `acceptance_rate=0.0238` (2.38 %) and `percent_changed=12.4 %`; `vol-ff5mom` lands at `acceptance_rate=0.0226` with `percent_changed=13.2 %`.
  - Flip-set files (`reports/rc-20251113/*/dm_flip_only.csv`) carry nontrivial coverage (`n_effective`≈110–117). The FF5+MOM run yields significant sign-test wins (EW vs baseline: z≈5.57, p≈9.3e‑10; MV vs baseline: z≈3.36, p≈9.8e‑4) while the off run remains neutral (stats = NaN but `n_effective=110`).
  - `reports/rc-20251113/vol-ff5mom/prewhiten_effect.csv` quantifies the paired delta: detection_rate +3.6 bps, ΔMSE(EW)=+4.1×10⁻¹¹, ΔMSE(MV)=−3.0×10⁻¹², ES95 errors tighten by ≈0.88 bps, and the sign-test p-values above corroborate the flip-set improvement.
  - Run manifest + provenance live in `reports/rc-20251113/run_manifest.json`, and the AWS logs (`reports/runs/20251112T232711Z/`, `reports/runs/20251113T014417Z/`) capture the exact commands, git SHA (`372d9d1cf4fe…`), dataset hashes, and telemetry for audit.
- **Next Actions**: None pending for this ticket; future tuning can explore softer δ-frac sweeps or factor toggles now that the reporting/artifact plumbing is in place.

## 2025-11-02T06:01Z
- **Step**: Implemented observed-factor prewhitening utilities with FF5/MKT fallback loader and regression residual outputs.
- **Decisions**: Normalised factor files via alias map and scaled percent inputs; fell back to equal-weight MKT proxy when datasets missing; exposed betas/intercepts/R² via `PrewhitenResult`.
- **Checks**: `pytest tests/baselines/test_prewhiten.py` (includes null FPR guardrail) passes.
- **Next Actions**: Tighten overlay substitution/shrinkage logic with deterministic seeds and calibration-aligned a-grid.

## 2025-11-02T06:05Z
- **Step**: Added overlay config with tyler edge default, deterministic gating, and selective eigenvalue substitution backed by shrinker baseline.
- **Decisions**: Defaulted to RIE-style shrinkage while allowing LW/OAS overrides; cap detections via `q_max`/`max_detections` and preserve Rayleigh quotient when swapping eigenvalues.
- **Checks**: `pytest tests/fjs/test_overlay.py` passes.
- **Next Actions**: Extend evaluation harness with regime splits, ES(95), diagnostics, and ETF alt-panel outputs.

## 2025-11-02T06:14Z
- **Step**: Built daily evaluation harness with ES(95) + calm/crisis splits, added diagnostics CSV/PNGs, and wired an ETF alt-panel wrapper.
- **Decisions**: Aggregated ΔMSE/DM against shrinker baseline, stored detection metrics per regime, and provided ETF CLI that reuses the evaluation pipeline.
- **Checks**: `pytest tests/experiments/test_eval_run.py` passes.
- **Next Actions**: Refresh docs, regenerate RC artifacts/galleries, and document overlay on/off diagnostics.

## 2025-11-02T06:32Z
- **Step**: Updated README/AGENTS with daily evaluation + ETF workflow, generated sample RC artifacts (`reports/rc-20251103/`), and added overlay toggle notes.
- **Decisions**: Defaulted evaluation `a_grid=60` for deterministic runtime; fallback loader handles wide/long returns; overlay eigenvalue failures revert to baseline shrinker.
- **Checks**: `python experiments/eval/run.py --returns-csv reports/rc-20251103/sample_returns.csv --window 40 --horizon 10 --out reports/rc-20251103/` succeeds.
- **Next Actions**: Run full test suite and prepare Conventional Commit summary.

## 2025-11-02T07:05Z
- **Step**: Added config-layer precedence (defaults → thresholds.json → YAML → CLI) with resolved-config echo, wrote `resolved_config.json`, and embedded reason-code enums in diagnostics.
- **Decisions**: Normalised CLI/YAML aliases (`--out` → `out_dir`), defaulted calm/crisis quantiles via layered config, and captured per-regime reason modes alongside resolved-config paths.
- **Checks**: `pytest tests/experiments/test_eval_run.py -q`.
- **Next Actions**: Guard volatility regime quantiles against look-ahead and centralise deterministic seeds/a-grid handling.

## 2025-11-02T07:24Z
- **Step**: Hardened volatility regime splits with train-only quantiles and past-only EWMA, centralised overlay seed/a-grid in config, and seeded numpy/random for deterministic runs.
- **Decisions**: Shifted EWMA by one day before lookups, wrote helper `_vol_thresholds` for tests, and recorded calm/crisis thresholds plus vol signals in diagnostics.
- **Checks**: `pytest tests/experiments/test_eval_run.py -q`.
- **Next Actions**: Tag slow tests, add CLI worker flag, and wire calibration cache controls for performance improvements.

## 2025-11-02T07:46Z
- **Step**: Marked heavy synthetic tests as slow with CI default `-m "not slow"`, added synthetic calibration caching with `_meta`+mtime guard plus `--force`, and introduced optional evaluation `--workers` using thread pooling.
- **Decisions**: Normalised config hashes via `calibration_cache_meta`, reused cached JSON when dependencies unchanged, and ensured parallel windows reuse sequential logic for identical outputs.
- **Checks**: `pytest tests/synthetic/test_harness_utils.py -q`, `pytest tests/experiments/test_eval_run.py -q`.
- **Next Actions**: Enhance turnover-aware MV, DM effective sample reporting, and add optional bootstrap scaffolding.

## 2025-11-02T08:18Z
- **Step**: Added turnover-aware MV weights (`mv_gamma`, `mv_tau`) with reproducible worker gating, aligned DM samples with effective n, and introduced optional block-bootstrap ΔMSE bands.
- **Decisions**: Stored per-window IDs for alignment, surfaced `n_effective` in DM outputs, and seeded bootstrap draws via `config.bootstrap_samples` for deterministic CI bands.
- **Checks**: `pytest tests/experiments/test_eval_run.py -q`, `pytest -q` (defaults exclude slow).
- **Next Actions**: Extend memo/gallery diagnostics and draft advisor RC brief with README linkage.

## 2025-11-02T08:46Z
- **Step**: Expanded reporting stack with memo reason-code tables, diagnostics plots (edge hist, isolation bars, stability scatter), and added `tools/build_brief.py` for the advisor one-pager plus README hooks.
- **Decisions**: Emitted per-window diagnostics to `diagnostics_detail.csv`, aggregated reason shares into markdown, and saved summary plots under `figures/<gallery>/summary/` for reuse.
- **Checks**: `pytest -q`; manual smoke of `tools/build_brief.py --config experiments/equity_panel/config.rc.yaml`.
- **Next Actions**: Coordinate gallery updates with upcoming RC sweep and validate advisor brief feedback loop.

## 2025-11-02T09:30Z
- **Step**: Added RC summary consolidation (`tools/make_summary.py`) with kill criteria + limitations artefacts, wired deterministic ablation runner (`experiments/ablate/run.py`) into `make rc`, and refreshed memo/brief templates to ingest summary/perf/detection CSVs.
- **Decisions**: Auto-discover latest `reports/rc-*/summary/`, derived kill checks (ΔMSE, detection bandwidth, alignment cosine, reason-code) into JSON + markdown, and surfaced global ablation deltas alongside updated bullets referencing reason codes, margins, and stability.
- **Checks**: `pytest tests/tools/test_make_summary.py tests/experiments/test_ablate_run.py tests/experiments/test_eval_run.py`.
- **Next Actions**: Run full `make rc` once summary/ablation cache warms and circulate updated memo/brief for advisor review.

## 2025-11-03T19:49Z
- **Step**: Added shrinker zero-fill WARN/PSD guards, introduced calm/crisis window sampling with a tiny ablation grid, and ran `make rc ABLA_GRID=experiments/ablate/ablation_matrix_tiny.yaml RC_PROGRESS=1 RC_WORKERS=9` to refresh RC artefacts (memo + brief rebuilt).
- **Decisions**: Sampled 10 calm windows uniformly and kept top-25 crisis windows by edge margin; accepted zero detections/ΔMSE (kill criteria fail) as signal that sampling is too aggressive and needs tuning before sign-off.
- **Checks**: `pytest tests/test_shrinkage.py tests/experiments/test_eval_run.py tests/experiments/test_ablate_run.py`, inspected `reports/rc-20251103/summary/kill_criteria.json`, regenerated `reports/memo.md` and `reports/brief.md`.
- **Next Actions**: Loosen calm/crisis limits or revisit overlay gating so detection coverage returns to target band ahead of advisor review.

## 2025-11-03T20:07Z
- **Step**: Created feature and docs branches (`feat/daily-groups-dow`, `feat/daily-groups-volstate`, `feat/rie-ewma`, `feat/prewhiten-overlay`, `feat/calibration-artifacts`, `docs/roadmap`) to stage sprint deliverables; verified ROADMAP.md already present.
- **Commands**: `git branch feat/daily-groups-dow`, `git branch feat/daily-groups-volstate`, `git branch feat/rie-ewma`, `git branch feat/prewhiten-overlay`, `git branch feat/calibration-artifacts`, `git branch docs/roadmap`, `git branch --list 'feat/*' 'docs/*'`.
- **Outputs**: Local branches available for sequential workstreams; documentation assets unchanged pending feature updates.
- **Checks**: `pytest -q`.
- **Next Actions**: Checkout `feat/daily-groups-dow` and implement DoW grouping module, tests, and smoke run.

## 2025-11-03T20:36Z
- **Step**: Implemented DoW daily grouping in `experiments/equity_panel/run.py` with grouped MANOVA prep, CLI design choices (`dow`/`vol`), and logging; expanded grouping tests to cover 3y replicates.
- **Commands**: `pytest tests/experiments/test_daily_grouping.py -q`, `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml --design dow --no-progress --assets-top 80 --stride-windows 4 --estimator oas`, `pytest -q`.
- **Outputs**: `experiments/equity_panel/outputs/` (dow smoke artifacts including edge_diag_window*.csv, spectrum plots, config_resolved.yaml).
- **Checks**: `pytest -q`.
- **Next Actions**: Switch to `feat/daily-groups-volstate`, add realized-vol grouping with tests, and rerun the daily smoke.

## 2025-11-05T21:55Z
- **Step**: Added synthetic ROC harness (`null.py`/`power.py`), enriched detection summaries with edge bands + MV solver telemetry, and locked MV defaults (ridge=1e-4, box [0,0.1], 5bps turnover) with CI/make plumbing.
- **Decisions**: Emit `roc_null.png`/`roc_power.png` + `calibration_defaults.json`, record `edge_tyler`, `edge_band_min/max`, gating/mv condition flags per window, and expose make targets (`env`, `run:equity_smoke`, `sweep:acceptance`) used by the smoke workflow.
- **Checks**: `make sweep:acceptance`, `make run:equity_smoke`, `make memo`, `make test`.
- **Next Actions**: Tidy README sections on calibration sweep, monitor CI smoke runtime, and iterate on acceptance thresholds as WRDS data shifts.
