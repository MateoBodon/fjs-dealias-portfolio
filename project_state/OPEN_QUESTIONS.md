# Open Questions

- **Nested design robustness**: Previous nested smoke runs showed 0% accepted detections due to guardrails; latest RC-lite omitted nested. Which guardrail (isolation, stability, off-component leak, q_max) is most binding, and what FPR/power trade-off is acceptable for nested Year⊃Week?
- **Crisis regime performance**: Crisis 2020 slices show de-aliased ΔMSE worse than shrinkage despite abundant detections. Should crisis runs use softer gating (soft mode, lower delta_frac) or alternative baselines (Tyler/POET) to avoid over-substitution?
- **Ablation coverage**: `config.ablation.smoke.yaml` currently times out; which parameters are highest value to keep? Can grid be pruned or parallelised without losing insight?
- **Edge calibration interaction**: δ_frac tables calibrated on SCM energy floors; how sensitive are Tyler/Huber edge modes to the same thresholds? Should calibration be edge-mode specific with updated ROC sweeps?
- **Alignment thresholds**: q2 alignment (`VOL_Q2_ALIGNMENT_MIN_COS`, `alignment_top_p`) is hand-tuned; what’s the empirical effect on FPR/power and substitution rates across regimes?
- **Prewhitening choices**: FF5+MOM is default; how do results shift under MKT-only or no prewhiten, especially for vol-state grouping? Is factor quality tracked sufficiently (R² histograms only)?
- **Cache trust**: Per-window caches keyed by code signature + data hashes; do code signature targets cover all relevant files (e.g., evaluation/report changes)? Risk of stale cached stats influencing reruns.
- **Unimplemented pieces**: Balanced weight computation and MP PDF edges are placeholders; do we need them for future papers?
- **Throughput vs determinism**: EXEC_MODE=throughput changes worker scaling and thread caps; how much bias/noise does this add to detection/DM statistics compared to deterministic runs?
