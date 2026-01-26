# Results (ticket-05)

## Notes
- Pre-run cleanup: stashed local changes with `git stash push -u -m "local pre ticket-05"` to keep non-code edits local.

## Deterministic smoke (rc-lite-sanity)
- rc_dir: reports/rc-20251221-sanity-20251221_220443
- overlay_forensics.csv rows (wc -l): 215
- limitations.md warnings:

```
## Excluded smoke-only runs (capped)
- reports/rc-20251221-sanity-20251221_220443/dow-tyler (cap_sources: date_truncation)
- reports/rc-20251221-sanity-20251221_220443/vol-tyler (cap_sources: date_truncation, window_coverage)

## Other limitations
- EW ΔMSE must not exceed baseline: value unavailable.
- MV ΔMSE must not exceed baseline: value unavailable.
- Detection coverage within target band: value unavailable.
- Average edge margin positive: value unavailable.
- Alignment cosine above 0.9: value unavailable.
- Overlay forensics: see summary/overlay_forensics.csv for changed-window diagnostics and loss deltas.
```

## Headline run attempt (daily DoW, uncapped target)
- rc_dir: reports/rc-ticket-05-20251221_221902
- run_dir: reports/rc-ticket-05-20251221_221902/dow-paper-v1
- summary outputs:
  - reports/rc-ticket-05-20251221_221902/summary/summary_perf.csv (empty; 0 rows)
  - reports/rc-ticket-05-20251221_221902/summary/summary_detection.csv (empty; 0 rows)
  - reports/rc-ticket-05-20251221_221902/summary/overlay_forensics.csv (wc -l: 6997)
  - reports/rc-ticket-05-20251221_221902/summary/limitations.md (see below)
  - reports/rc-ticket-05-20251221_221902/summary/completeness.json

### Acceptance checks (FAILED)
- run.json windows:
  - cap_active: True
  - cap_sources: ['window_coverage']
  - window_coverage: 0.9682933553901296
  - windows_evaluated: 3512 / windows_requested: 3627
- summary_perf.csv empty -> comparison_valid_* and n_effective_* unavailable.
- limitations.md:

```
## Excluded smoke-only runs (capped)
- reports/rc-ticket-05-20251221_221902/dow-paper-v1 (cap_sources: window_coverage)

## Other limitations
- EW ΔMSE must not exceed baseline: value unavailable.
- MV ΔMSE must not exceed baseline: value unavailable.
- Detection coverage within target band: value unavailable.
- Average edge margin positive: value unavailable.
- Alignment cosine above 0.9: value unavailable.
- Overlay forensics: see summary/overlay_forensics.csv for changed-window diagnostics and loss deltas.
```

### Diagnostic notes (coverage failure)
- diagnostics_detail reason_code counts: accepted=3512, holdout_empty=115 (matches missing windows).
- design_ok==0 windows all have holdout_empty and NaN window_start/window_id.

Status: stop-the-line. Headline-eligible run not achieved due to window_coverage cap.
## Bundle
- docs/gpt_bundles/20251222_004303_ticket-05_20251221_220252_ticket-05_advisor-ready-rc.zip

## Follow-ups
- PROGRESS.md / project_state/CURRENT_RESULTS.md / docs/CODEX_SPRINT_TICKETS.md not updated because headline eligibility failed (cap_active true).

## Checklist (ticket-05 requirements)

### Process / logs
- Run log dir present: PASS (`docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/` contains PROMPT/COMMANDS/RESULTS/TESTS/META).
- Commits include “Tests run:” in body: N/A (no commits made for this ticket).
- DIFF.patch: N/A (no git changes to patch; reproducible from git).

### Validity (headline evidence)
- reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json cap_active=false, cap_sources=[]: FAIL (cap_active true; cap_sources ["window_coverage"]).
- summary/summary_perf.csv comparison_valid_mse=1 and comparison_valid_qlike=1 (and DM if used): FAIL (summary_perf.csv empty).
- n_effective_mse>=50 and n_effective_qlike>=50: FAIL (summary_perf.csv empty).
- summary/limitations.md has no excluded-headline-run section for caps or MV skip: FAIL (limitations.md includes excluded capped run section).

### Required artifacts
- summary/overlay_forensics.csv exists + rows: PASS (`reports/rc-ticket-05-20251221_221902/summary/overlay_forensics.csv`, 6997 rows).
- summary/completeness.json exists: PASS (`reports/rc-ticket-05-20251221_221902/summary/completeness.json`).
- summary/summary_detection.csv exists (detection rate present): FAIL (file exists but empty; no detection rate rows).

### Docs updates
- PROGRESS.md updated: FAIL (not updated; headline eligibility failed).
- project_state/CURRENT_RESULTS.md updated: FAIL (not updated; headline eligibility failed).

### No fake fixes
- No always-reject/always-accept gating hacks: PASS (no changes made).
- No disabling overlay without explicit toggle + documentation: PASS (no changes made).
- No hidden caps (--max-windows / date truncation) for headline runs: FAIL (cap from window_coverage; run excluded as capped).
