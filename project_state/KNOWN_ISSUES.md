# Known Issues / Limitations

- **Nested coverage fragile**: Guardrails tightened (use_tvector on, eps≈1, off_leak=0.3) now eliminate null acceptance but also make nested RC/smoke highly conservative. Latest smoke (`experiments/equity_panel/outputs_nested_smoke_postfix`) shows 0 detections across 10 windows; moderate synthetic power is only ~0.42 (post_smoke_20251218), so tuning may still be needed to balance recall vs FPR.
- **Nested smoke still 0% after relax**: Latest deterministic smoke run remains at 0 detections with skip_reason `no_isolated_spike` despite calibrated delta entries; monitor after any further tuning and consider softer eta/delta_frac if recall is required.
- **Crisis degradation**: Crisis 2020 runs show overlay worse than shrinkage (ΔMSE > 0, significant DM p-values); current gating may be too permissive during stress regimes.
- **Ablation grid timing out**: `config.ablation.smoke.yaml` often fails to finish; gallery shows placeholder ablation section.
- **PSD clipping hides problems**: Overlay covariance enforces PSD by clipping negative eigenvalues/ridge; may mask unstable detections instead of surfacing warnings.
- **Cache staleness risk**: Window cache keys exclude report/evaluation code; cached stats may become inconsistent after logic changes unless cache dir cleared.
- **Optional dependencies**: `cvxpy` required for exact min-var; absence silently falls back to equal-weight with `converged=False`. `matplotlib` optional; plots skipped when missing.
- **Registry hash tolerance**: `data.registry.assert_registered_dataset` allows drift for `data/returns_daily.csv` in canonical repo; could hide accidental data changes if not checked.
- **Unimplemented functions**: `balanced.compute_balanced_weights` and `evaluation.marchenko_pastur_edges/pdf` are stubs; calling them raises NotImplementedError.
- **Threading variability**: EXEC_MODE throughput increases BLAS threads; non-deterministic ordering may affect marginal stats.
- **rc-lite-sanity kill failures**: Latest rc-lite-sanity (2025-12-09 stamp) fails kill criteria because ΔMSE>0 while percent_changed≈100%; DM stats empty. Summary_sanity now explicitly flags DoW + vol overlay_effect = harmful.
- **Weekly Dec 2025 smoke = 0 detections**: Both DoW and nested weekly runs in rc-lite-sanity report detection_windows=0 despite calibrated gates and relaxed thresholds.
- **Partial RC dir**: `reports/rc-20251208/` contains only resolved_config/prewhiten files; run appears incomplete yet is indistinguishable from a completed RC in discovery scripts.
