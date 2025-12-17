# Roadmap

## Short-term (1–2 weeks)
- Debug rc-lite-sanity: reduce percent_changed and eliminate positive ΔMSE (check gate/Δ aggregation, coarse-candidate effect); incorporate vol-state run into summary/kill_criteria.
- Investigate December weekly smoke/nested 0 detections; test loosened η/δ_frac/energy_min_abs and use_tvector toggles on p≈188, T≈60–80.
- Finish ablation grid regeneration (tiny matrix) and re-enable gallery ablation section; ensure make rc-ablations completes on Hetzner.
- Add guard in make_summary/summarize_rc_sanity to flag partial runs (e.g., reports/rc-20251208).
- Refresh README/REPORT “Current Status” to point at rc-lite-sanity 20251209 results.

## Medium-term (3–5 weeks)
- Calibrate edge thresholds for Tyler/Huber modes separately and extend p×t coverage to nested regimes; rerun synthetic sweeps.
- Add deterministic `nested-sanity` target with short windows and emit summary comparable to rc-lite-sanity.
- Improve cache versioning (include evaluation/report hashes) and optional `--no-cache` flag in equity_panel.
- Harden summary tooling: detect missing vol-state summaries, warn on incomplete RC dirs, optionally combine daily + weekly outputs.
- Expand flip-set diagnostics (delta_frac_used, alignment cos histograms) in daily/weekly outputs.

## Long-term (5–10 weeks)
- Implement balanced weight computation and MP PDF utilities (stubs).
- Broaden factor support (macro/industry), add factor quality dashboards, and run systematic prewhitening comparisons.
- Pipeline hardening: CI for `make rc-lite-sanity` subset; automated memo/brief upload; run discovery that filters incomplete runs.
- Research extensions: adaptive q_max per regime, turnover-aware overlay, robust edges on real panels, nested design power study with richer synthetic models.

## Dependencies / sequencing
- Calibrate edges before nested/weekly retuning; nested tuning depends on updated thresholds and cache hygiene.
- Finish ablation grid + summary fixes before regenerating gallery/memo.
- Cache/versioning changes should precede new RC/rc-lite-sanity drops to avoid mixing old/new stats.
