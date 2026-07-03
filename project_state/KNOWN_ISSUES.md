---
generated: 2026-02-16T02:32:01Z
git_sha: 1371b3c2e7197c3629cc20e4e67c1f435f3ca13a
git_branch: codex/ticket-27-repo-hygiene-cleanup
commands:
  - manual documentation recenter for ticket-31
  - artifact verification from reports/* files
---
# Known Issues

- **T-012 is recovered but not cleanly ratified**: the four-leg daily DoW matrix artifacts were recovered and appear scientifically usable, but the original T-012 review failed because the long-run monitoring/audit trail was not fully preserved.
- **Daily DoW remains empirical-only**: T-008/T-010/T-012 support an empirical lane, but they do not provide detector validation or recover the clean weekly oneway / FJS theory story.
- **Heavy T-012 details are local-only**: the full recovered tree is under `/Volumes/Storage/Projects/fjs/_recovery/recovered_artifacts/rc-t-012`; Git tracks only curated summary surfaces under `docs/artifacts/rc-t-012/`.
- **Injection sensitivity is flat-zero on current week-design evidence**: `reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv` shows detection/acceptance at 0.0 across tested `mu` values.
- **Advisor-ready uncapped headline run still pending**: no updated week-design run has yet closed the gate for advisor-safe headline reporting with meaningful effect evidence.
- **Detection diagnostics mismatch remains unresolved**: pre-gate reasons in `reports/inject_spike/20251226_ticket24_week_full_fix/gating_reasons.csv` are dominated by `tvec_off_component` and no-root buckets, which blocks interpretation.
- **Nested design remains secondary**: calibration coverage improved, but tiny real-data nested smoke still has zero detections and skip reasons dominated by stability/no-isolated-spike, so nested is not currently a headline path.
