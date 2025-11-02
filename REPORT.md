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
- **Checks**: `pytest tests/test_power_null.py -q -m slow`, `pytest tests/experiments/test_eval_run.py -q`.
- **Next Actions**: Enhance turnover-aware MV, DM effective sample reporting, and add optional bootstrap scaffolding.
