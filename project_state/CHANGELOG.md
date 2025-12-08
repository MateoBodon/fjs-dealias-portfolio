# Changelog (reconstructed)

- **2025-12-08** — README/RUNBOOK updated with latest RC-lite status, data hashes, calibration defaults; RUNBOOK documents Hetzner workflow.
- **2025-11-21** — RC-lite (DoW/Vol, Tyler edge, FF5+MOM, capped 200 windows) produced `reports/rc-20251121/`; calibration sweep (HARNESS_TRIALS=800, deterministic) refreshed `calibration_defaults.json` and `calibration/edge_delta_thresholds.json`; memo/brief regenerated.
- **2025-11 (early)** — Added robust edge modes (Tyler/Huber) and gating telemetry; observed-factor & POET-lite baselines integrated into evaluation; daily eval gained flip-set DM and prewhiten effect tooling; overlay soft/strict gate options expanded.
- **2025-10** — Gallery/memo/brief pipeline consolidated (`tools/build_gallery.py`, `build_memo.py`, `build_brief.py`), RC make targets orchestrate smoke/nested/crisis batches; per-window cache + code signatures introduced.
- **2025-09** — Factor registry + data registry validation tightened; synthetic harness sharding/reduction (`tools/shard_grid.py`, `reduce_calibration.py`) added; evaluation thresholds JSON introduced.
- **2025-08 and earlier** — Core MANOVA detection (`fjs.dealias`, `fjs.mp`), balanced/nested stats, shrinkage baselines, and synthetic one-way benchmarks established; initial tests and reporting fixtures landed.

(Changelog is partial; see `PROGRESS.md` and RC manifests for run-level history.)
