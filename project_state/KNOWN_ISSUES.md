# Known Issues / Limitations

- **Nested coverage fragile**: Guardrails (isolation, stability_eta, off-component cap) can zero out nested detections; nested RC not present in latest drop. Needs re-tuning.
- **Nested smoke still 0% after relax**: Even with relaxed delta_frac/eta and non-isolated fallback, nested smoke runs (p≈188, T≈60–80) record 0 acceptances; calibrated delta entries missing for these (p, T) pairs fall back to config values. Additional diagnostics now logged (nested_years/weeks, prep events), but gating/detection logic needs further tuning.
- **Crisis degradation**: Crisis 2020 runs show overlay worse than shrinkage (ΔMSE > 0, significant DM p-values); current gating may be too permissive during stress regimes.
- **Ablation grid timing out**: `config.ablation.smoke.yaml` often fails to finish; gallery shows placeholder ablation section.
- **PSD clipping hides problems**: Overlay covariance enforces PSD by clipping negative eigenvalues/ridge; may mask unstable detections instead of surfacing warnings.
- **Cache staleness risk**: Window cache keys exclude report/evaluation code; cached stats may become inconsistent after logic changes unless cache dir cleared.
- **Optional dependencies**: `cvxpy` required for exact min-var; absence silently falls back to equal-weight with `converged=False`. `matplotlib` optional; plots skipped when missing.
- **Registry hash tolerance**: `data.registry.assert_registered_dataset` allows drift for `data/returns_daily.csv` in canonical repo; could hide accidental data changes if not checked.
- **Unimplemented functions**: `balanced.compute_balanced_weights` and `evaluation.marchenko_pastur_edges/pdf` are stubs; calling them raises NotImplementedError.
- **Threading variability**: EXEC_MODE throughput increases BLAS threads; non-deterministic ordering may affect marginal stats.
- **rc-lite-sanity kill failures**: Latest rc-lite-sanity (2025-12-09 stamp) fails kill criteria because ΔMSE>0 while percent_changed≈100%; DM stats empty. Indicates over-substitution or sign mismatch in ΔMSE aggregation.
- **Vol-state summary missing**: make_summary output for rc-lite-sanity includes only DoW design; vol-state metrics remain stranded in run dirs.
- **Weekly Dec 2025 smoke = 0 detections**: Both DoW and nested weekly runs in rc-lite-sanity report detection_windows=0 despite calibrated gates and relaxed thresholds.
- **Partial RC dir**: `reports/rc-20251208/` contains only resolved_config/prewhiten files; run appears incomplete yet is indistinguishable from a completed RC in discovery scripts.
