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
- **rc-lite-sanity ΔMSE**: Daily rc-lite-sanity fails kill criteria because ΔMSE>0 despite moderate detection and perfect alignment—should gating reduce percent_changed, or is baseline comparison sign flipped?
- **Vol-state summary omission**: rc-lite-sanity summary/kill_criteria include only DoW; is this a tool bug (make_summary) or intentional filter? Need parity across designs.
- **Detection vs percent_changed**: Both DoW and vol runs show percent_changed ≈94–100% while detection_rate ≈5%. Is substitution applied to all windows once any detection exists? Verify gating logic in overlay/eval runner.
- **Weekly detection drought**: Why do December weekly DoW/nested smoke runs still report 0 detections (p≈188, window 52/6)? Are balanced-week filters too strict or edges too conservative at tyler? Should we re-enable use_tvector or loosen energy_min_abs?
- **Data loader mismatch**: `experiments/eval/run.py` imports `data.loader` but falls back to inline loader; should we formalize a shared `data.loader` to avoid divergent behavior?
- **rc-20251208 incomplete run**: `reports/rc-20251208/` lacks metrics—was the run interrupted or filtered out? Should make_summary detect and warn on partial runs?
