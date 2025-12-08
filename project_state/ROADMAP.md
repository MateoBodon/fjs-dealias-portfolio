# Roadmap

## Short-term (1–2 weeks)
- Rerun RC-lite without 200-window cap and include nested design; verify detection 2–6% band and DM coverage.
- Trim/parallelise ablation grid so `ablation_summary.csv` regenerates; re-enable gallery ablation section.
- Refresh calibration sweeps for Tyler/Huber edge modes (HARNESS_TRIALS≈400–800) and update `calibration_defaults.json`/`edge_delta_thresholds.json`.
- Document Hetzner execution profile in `docs/HPC.md` and ensure `make test-fast` + smoke runs green on both local and Hetzner.

## Medium-term (3–5 weeks)
- Tune nested guardrails (alignment, isolation, q_max) using synthetic nested null/power + WRDS nested runs; add `make nested-sanity` target with compact summary.
- Improve crisis handling: experiment with softer gating or alternate baselines (Tyler/POET) and add crisis-specific memo badges.
- Strengthen caching hygiene (include evaluation/report hashes in signature; add `--no-cache` option or cache versioning).
- Expand flip-set diagnostics (e.g., histograms of delta_frac_used, alignment cos) in daily/weekly outputs.

## Long-term (5–10 weeks)
- Implement balanced weight computation and MP PDF utilities (stubs in `balanced.py`, `mp.py`).
- Broaden factor support (e.g., macro/industry) and assess prewhitening impact systematically; add factor quality dashboards.
- Pipeline hardening: CI job for `make rc-lite-sanity` on representative subset, automated memo/brief artifact upload.
- Research extensions: evaluate huber/tyler robust edges on real panels, explore adaptive q_max per regime, investigate turnover-aware overlay.

## Dependencies / sequencing
- Calibrate edges before retuning guardrails; nested tuning depends on updated thresholds.
- Finish ablation grid improvements prior to memo/gallary refresh to avoid missing sections.
- Cache/versioning changes should precede large-scale reruns to avoid mixing old/new stats.
