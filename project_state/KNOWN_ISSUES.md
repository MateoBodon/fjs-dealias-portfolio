---
generated: 2026-02-16T02:32:01Z
git_sha: 1371b3c2e7197c3629cc20e4e67c1f435f3ca13a
git_branch: codex/ticket-27-repo-hygiene-cleanup
commands:
  - manual documentation recenter for ticket-31
  - artifact verification from reports/* files
---
# Known Issues

- **Injection sensitivity is flat-zero on current week-design evidence**: `reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv` shows detection/acceptance at 0.0 across tested `mu` values.
- **Advisor-ready uncapped headline run still pending**: no updated week-design run has yet closed the gate for advisor-safe headline reporting with meaningful effect evidence.
- **Detection diagnostics mismatch remains unresolved**: pre-gate reasons in `reports/inject_spike/20251226_ticket24_week_full_fix/gating_reasons.csv` are dominated by `tvec_off_component` and no-root buckets, which blocks interpretation.
- **Nested design remains secondary**: calibration coverage improved, but tiny real-data nested smoke still has zero detections and skip reasons dominated by stability/no-isolated-spike, so nested is not currently a headline path.
