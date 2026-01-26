# Results

- Added pre-gate telemetry CSV emission for inject-spike with per-window tvec diagnostics and scalar checks.
- Default inject-mode set to between for CLI and Makefile targets.
- Added unit coverage for telemetry CSV writer helper.
- Smoke (sample_spike, 76 windows): baseline_detect=0.211 baseline_accept=0.000; μ=5/10/20 det=1.00 acc=1.00. Output: `reports/inject_spike_smoke/20260126_043000_ticket-028_inject-spike-smoke/`.
- Acceptance smoke (sample, max_windows=1): baseline_detect=0.000 baseline_accept=0.000; μ=50 det=1.00 acc=1.00. Output: `reports/inject_spike_smoke/20260126_045000_ticket-028_inject-spike-smoke-mini/`.
