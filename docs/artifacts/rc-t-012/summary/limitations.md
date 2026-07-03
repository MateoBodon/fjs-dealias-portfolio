## Conditional reporting (changed-window only)
Δ metrics are computed on aligned overlay/baseline windows with changed_flag=1.
- full/ew: n_changed=1.75e+03, changed_frac=1
- full/mv: n_changed=1.75e+03, changed_frac=1
- calm/ew: n_changed=715, changed_frac=1
- calm/mv: n_changed=715, changed_frac=1
- crisis/ew: n_changed=1.03e+03, changed_frac=1
- crisis/mv: n_changed=1.03e+03, changed_frac=1
- full/ew: n_changed=1.69e+03, changed_frac=1
- full/mv: n_changed=1.69e+03, changed_frac=1
- calm/ew: n_changed=608, changed_frac=1
- calm/mv: n_changed=608, changed_frac=1
- crisis/ew: n_changed=1.01e+03, changed_frac=1
- crisis/mv: n_changed=1.01e+03, changed_frac=1
- full/ew: n_changed=1.8e+03, changed_frac=1
- full/mv: n_changed=1.8e+03, changed_frac=1
- calm/ew: n_changed=867, changed_frac=1
- calm/mv: n_changed=867, changed_frac=1
- crisis/ew: n_changed=835, changed_frac=1
- crisis/mv: n_changed=835, changed_frac=1
- full/ew: n_changed=1.67e+03, changed_frac=1
- full/mv: n_changed=1.67e+03, changed_frac=1
- calm/ew: n_changed=827, changed_frac=1
- calm/mv: n_changed=827, changed_frac=1
- crisis/ew: n_changed=811, changed_frac=1
- crisis/mv: n_changed=811, changed_frac=1

## Other limitations
- EW ΔMSE must not exceed baseline: observed 2.64e-11 vs threshold {'max': 0.0}.
- reports/rc-t-012/dow-paper-v1_ff5mom_w126: windows dropped from planning (holdout_empty: 115).
- reports/rc-t-012/dow-paper-v1_ff5mom_w252: windows dropped from planning (holdout_empty: 183).
- reports/rc-t-012/dow-paper-v1_noprewhiten_w126: windows dropped from planning (holdout_empty: 115).
- reports/rc-t-012/dow-paper-v1_noprewhiten_w252: windows dropped from planning (holdout_empty: 183).
- overlay_forensics missing metrics_detail.csv in _preserved_interrupted_20260401_000435
- overlay_forensics missing diagnostics_detail.csv in _preserved_interrupted_20260401_000435
- Overlay forensics: see summary/overlay_forensics.csv for changed-window diagnostics and loss deltas.