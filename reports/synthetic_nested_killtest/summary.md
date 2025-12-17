# Nested synthetic kill-test

- **tyler | moderate**: detection_rate=1.000 over 12 trials; skip_top=n/a; calib_missing_share=0.000
- **tyler | null**: detection_rate=1.000 over 12 trials; skip_top=n/a; calib_missing_share=0.000
- **tyler | strong**: detection_rate=1.000 over 12 trials; skip_top=n/a; calib_missing_share=0.000

Observed FPR is effectively 1.0 in the null scenario, indicating the current nested gating setup is unsafe (always accepts) even with calibrated delta_frac and higher absolute delta. Power is indistinguishable from the null because acceptance is unconditional.
