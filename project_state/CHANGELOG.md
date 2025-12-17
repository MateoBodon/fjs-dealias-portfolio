# Changelog (reconstructed)

- **2025-12-17** — Refreshed `project_state/` spine with rc-lite-sanity 20251209 findings; added INDEX.md; expanded module/function index, config reference, pipeline/dataflow docs; captured rc-lite-sanity kill-criteria failure and weekly detection drought; flagged partial rc-20251208 run and vol-state summary gap.
- **2025-12-09** — Added `make rc-lite-sanity` (daily DoW/vol + weekly DoW/nested smoke) with timestamped outputs and new `tools/summarize_rc_sanity.py`; nested smoke config relaxed (delta_frac/eta/non-isolated guardrails) and detection summaries now include nested counts/prep logging; nested acceptance still 0%, calibration missing for p≈188/T≈60–80 noted. rc-lite-sanity run at `reports/rc-20251208-sanity-20251209_001356`.
- **2025-12-08** — README/RUNBOOK updated with latest RC-lite status, data hashes, calibration defaults; RUNBOOK documents Hetzner workflow.
- **2025-11-21** — RC-lite (DoW/Vol, Tyler edge, FF5+MOM, capped 200 windows) produced `reports/rc-20251121/`; calibration sweep (HARNESS_TRIALS=800, deterministic) refreshed `calibration_defaults.json` and `calibration/edge_delta_thresholds.json`; memo/brief regenerated.
- **2025-11 (early)** — Added robust edge modes (Tyler/Huber) and gating telemetry; observed-factor & POET-lite baselines integrated into evaluation; daily eval gained flip-set DM and prewhiten effect tooling; overlay soft/strict gate options expanded.
- **2025-10** — Gallery/memo/brief pipeline consolidated (`tools/build_gallery.py`, `build_memo.py`, `build_brief.py`), RC make targets orchestrate smoke/nested/crisis batches; per-window cache + code signatures introduced.
- **2025-09** — Factor registry + data registry validation tightened; synthetic harness sharding/reduction (`tools/shard_grid.py`, `reduce_calibration.py`) added; evaluation thresholds JSON introduced.
- **2025-08 and earlier** — Core MANOVA detection (`fjs.dealias`, `fjs.mp`), balanced/nested stats, shrinkage baselines, and synthetic one-way benchmarks established; initial tests and reporting fixtures landed.

(Changelog is partial; see `PROGRESS.md` and RC manifests for run-level history.)
