# Roadmap

**Short-term focus (Dec 2025, next 1–2 sprints)**

- Treat `make rc-lite-sanity` as the main sanity RC for now. The latest run (2023H1, 50 assets) shows:
  - daily DoW detection ≈ 5.36% with ~100% flips and **ΔMSE>0** vs Ledoit–Wolf,
  - daily vol-state detection ≈ 5.22% with ~94% flips and no clear ΔMSE benefit,
  - weekly DoW and nested smokes with **0 accepted windows**.
- Short-term work therefore focuses on:
  - (i) completing Tyler/Huber edge calibration for p≈190, T≈60–80 and improving reason logging for empty detections,  
  - (ii) running a minimal nested synthetic kill-test to decide whether nested is viable at all, and  
  - (iii) making rc-lite-sanity summaries explicitly report when overlay harms risk (ΔMSE>0), so we do not misinterpret the current state.
- Nested is considered **guilty until proven useful**: if the calibration + synthetic kill-tests do not yield nontrivial coverage and some performance benefit, nested will be demoted to an exploratory side note rather than a main design in the eventual paper.


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
