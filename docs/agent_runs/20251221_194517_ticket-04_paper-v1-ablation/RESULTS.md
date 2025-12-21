# Results
- Implemented paper-v1 ablation runner:
  - New pinned config: `experiments/eval/config.paper_v1.yaml` (DoW design, min_comparison_windows=50, MV constraints, gating defaults). Overlay grid reduced to 30 to keep deterministic smoke runtime tractable.
  - New Make target: `rc-paper-v1-ablate` (SCM via `--shrinker sample`, OAS, RIE; overlay OFF uses `--q-max 0`).
  - Added `tools/paper_v1_ablation.py` + unit test; summary_perf now includes delta_qlike + comparison_valid_{mse,es,qlike} and multi-run summaries are per-run when rc dir contains sub-runs.
  - Eval runner short-circuits detections when `q_max <= 0` to make overlay OFF runs fast and explicit.

- Validation run (deterministic, small real dataset):
  - Returns subset: `reports/fixtures/returns_daily_small.csv` (196 dates, 60 tickers, 2024-03-22 → 2024-12-31).
  - Run root: `reports/rc-paper-v1-ablate-20251221_205751/`.
  - cap flags (from run.json):
    - scm_off/on, oas_off/on, rie_off/on all report `cap_active=false`, `cap_sources=[]`.
  - Summary artifacts:
    - `summary/summary_perf.csv` rows: 36
    - `summary/summary_detection.csv` rows: 18
    - `summary/paper_v1_ablation.csv` rows: 6
    - `summary/limitations.md` has no capped/mv-skip exclusions; notes insufficient aligned windows (n_effective < min_comparison_windows) for some comparisons.

- Notes on failed attempts:
  - First rc-paper-v1-ablate run failed on invalid shrinker `scm` (fixed by mapping SCM→`sample`).
  - Several early runs were interrupted during overlay ON runs; final run completed at `reports/rc-paper-v1-ablate-20251221_205751/`.
- Bundle: `docs/gpt_bundles/20251221_211157_ticket-04_20251221_194517_ticket-04_paper-v1-ablation.zip`
