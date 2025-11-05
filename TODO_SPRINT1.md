# Sprint-1 Checklist

- [ ] Replace missing `PLAN.md` context (confirm scope via Roadmap/RUNBOOK) and circulate once drafted.
- [x] Implement synthetic null/power harnesses under `experiments/synthetic/` with ROC outputs and metadata plumbing.
- [x] Calibrate acceptance thresholds (target FPR ≤ 2%, μ∈{4,6,8}) and persist to `calibration_defaults.json`.
- [x] Wire SCM/Tyler MP edge modes into runners and emit per-window `detection_summary.csv` with edges, stability, leakage.
- [x] Enforce MV defaults (ridge `1e-4`, box `[0, 0.1]`, turnover 5–10 bps) plus condition number guards and unit coverage.
- [x] Extend Make targets (`env`, `run:equity_smoke`, `sweep:acceptance`, `memo`) and add Github Actions smoke workflow.
- [x] Stand up `PROGRESS.md` entry with run SHA, dataset digest, ROC figures, and defaults.
- [x] Ensure memo/brief templates pick up ROC + acceptance defaults (update `reports/memo/` if present).
- [x] Run `make test` and `make run:equity_smoke`; archive generated artifacts under `reports/`.
- [ ] Prepare Conventional Commits and push once smoke is green.
