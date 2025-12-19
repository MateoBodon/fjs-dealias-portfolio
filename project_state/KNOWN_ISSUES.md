# Known Issues / Limitations

- **Nested coverage fragile**: Guardrails (isolation, stability_eta, off-component cap) can zero out nested detections; nested RC not present in latest drop. Needs re-tuning.
- **Nested smoke still 0% after relax**: Even with relaxed delta_frac/eta and non-isolated fallback, nested smoke runs (p≈188, T≈60–80) record 0 acceptances; calibrated delta entries now added for these (p, T) pairs but detection remains 0. Additional diagnostics now logged (nested_years/weeks, prep events), but gating/detection logic needs further tuning.
- **Crisis degradation**: Crisis 2020 runs show overlay worse than shrinkage (ΔMSE > 0, significant DM p-values); current gating may be too permissive during stress regimes.
- **Ablation grid timing out**: `config.ablation.smoke.yaml` often fails to finish; gallery shows placeholder ablation section.
- **PSD clipping hides problems**: Overlay covariance enforces PSD by clipping negative eigenvalues/ridge; may mask unstable detections instead of surfacing warnings.
- **Cache staleness risk**: Window cache keys exclude report/evaluation code; cached stats may become inconsistent after logic changes unless cache dir cleared.
- **Optional dependencies**: `cvxpy` required for exact min-var; absence silently falls back to equal-weight with `converged=False`. `matplotlib` optional; plots skipped when missing.
- **Registry hash tolerance**: `data.registry.assert_registered_dataset` allows drift for `data/returns_daily.csv` in canonical repo; could hide accidental data changes if not checked.
- **Unimplemented functions**: `balanced.compute_balanced_weights` and `evaluation.marchenko_pastur_edges/pdf` are stubs; calling them raises NotImplementedError.
- **Threading variability**: EXEC_MODE throughput increases BLAS threads; non-deterministic ordering may affect marginal stats.
- **Nested synthetic kill-test blows FPR**: Synthetic nested harness (p≈200, weeks 6–8, reps=5) shows FPR≈1.0 under null—overlay accepts every window even with calibrated delta_frac and delta=0.35—so nested gating is currently unsafe.
- **rc-lite-sanity kill failures**: Latest rc-lite-sanity (2025-12-09 stamp) fails kill criteria because ΔMSE>0 while percent_changed≈100%; DM stats empty. Summary_sanity now explicitly flags DoW + vol overlay_effect = harmful.
- **Weekly Dec 2025 smoke = 0 detections**: Both DoW and nested weekly runs in rc-lite-sanity report detection_windows=0 despite calibrated gates and relaxed thresholds.
- **Partial RC dir**: `reports/rc-20251208/` contains only resolved_config/prewhiten files; run appears incomplete yet is indistinguishable from a completed RC in discovery scripts. **Fixed 2025-12-19** via completeness checks in `tools/make_summary.py` / `tools/summarize_rc_sanity.py`; rc-lite-sanity 20251219 run logs completeness metadata and flags missing sections.
