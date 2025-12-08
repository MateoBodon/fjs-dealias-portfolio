# Known Issues / Limitations

- **Nested coverage fragile**: Guardrails (isolation, stability_eta, off-component cap) can zero out nested detections; nested RC not present in latest drop. Needs re-tuning.
- **Crisis degradation**: Crisis 2020 runs show overlay worse than shrinkage (ΔMSE > 0, significant DM p-values); current gating may be too permissive during stress regimes.
- **Ablation grid timing out**: `config.ablation.smoke.yaml` often fails to finish; gallery shows placeholder ablation section.
- **PSD clipping hides problems**: Overlay covariance enforces PSD by clipping negative eigenvalues/ridge; may mask unstable detections instead of surfacing warnings.
- **Cache staleness risk**: Window cache keys exclude report/evaluation code; cached stats may become inconsistent after logic changes unless cache dir cleared.
- **Optional dependencies**: `cvxpy` required for exact min-var; absence silently falls back to equal-weight with `converged=False`. `matplotlib` optional; plots skipped when missing.
- **Registry hash tolerance**: `data.registry.assert_registered_dataset` allows drift for `data/returns_daily.csv` in canonical repo; could hide accidental data changes if not checked.
- **Unimplemented functions**: `balanced.compute_balanced_weights` and `evaluation.marchenko_pastur_edges/pdf` are stubs; calling them raises NotImplementedError.
- **Threading variability**: EXEC_MODE throughput increases BLAS threads; non-deterministic ordering may affect marginal stats.
