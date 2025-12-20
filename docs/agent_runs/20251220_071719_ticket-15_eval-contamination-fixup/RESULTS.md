## Initial audit
$ git status -sb
## codex/ticket-15-eval-contamination-fixup
 M docs/CODEX_SPRINT_TICKETS.md
 M docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
?? docs/agent_runs/20251220_071719_ticket-15_eval-contamination-fixup/
?? reports/eval-ticket-11-smoke/
$ git rev-parse HEAD
13f76f830d8eacc105c7135387ae628e76aba819

$ git diff --stat
 docs/CODEX_SPRINT_TICKETS.md                                  | 11 +++++++++++
 .../20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md     |  6 ++++++
 2 files changed, 17 insertions(+)

$ git diff | head -n 200
diff --git a/docs/CODEX_SPRINT_TICKETS.md b/docs/CODEX_SPRINT_TICKETS.md
index adcdc89..ae336e8 100644
--- a/docs/CODEX_SPRINT_TICKETS.md
+++ b/docs/CODEX_SPRINT_TICKETS.md
@@ -1,3 +1,14 @@
+ticket-10: FAIL (no commits + empty DIFF.patch → not auditable/mergeable).
+
+Keep ticket-11 / ticket-12 in queue, but do not proceed until we can review and merge ticket-10’s claimed calibration plumbing.
+
+Next ticket to run (exactly one): add and run a fixup ticket
+
+Select: NEW ticket-14 — “Ticket‑10 Fixup: make nested calibration mergeable + auditable”
+
+Rationale: the technical result might be correct, but we can’t merge or trust it without committed diffs + a non-empty patch + a nested-specific smoke.
+
+
 # Codex Sprint Tickets (NEXT SPRINT ONLY)
 
 Ordering principle: unblock validity first, then calibrate/extend.
diff --git a/docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md b/docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
index 3c3fc77..8297168 100644
--- a/docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
+++ b/docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
@@ -68,3 +68,9 @@ PY
 - git add docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/RESULTS.md
 - git add docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
 - git rev-parse HEAD
+- git add AGENTS.md docs/CODEX_SPRINT_TICKETS.md docs/DOCS_AND_LOGGING_SYSTEM.md
+- git commit -m "Carry forward doc edits" -m "Tests: source .venv/bin/activate && make test-fast
+Smoke: source .venv/bin/activate && EXEC_MODE=deterministic make run:equity_smoke
+Run log: docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/"
+- git checkout main
+- git merge --no-ff ticket-10-nested-null-fpr
\n## Smoke run proof (reports/eval-ticket-15-smoke-aligned5)
$ head -n 5 $REPORT/full/metrics.csv (selected cols)
  regime estimator  ... comparison_valid_es  comparison_valid_qlike
0   full  baseline  ...                 NaN                     NaN
1   full  baseline  ...                 NaN                     NaN
2   full        cc  ...                 NaN                     NaN
3   full        cc  ...                 NaN                     NaN
4   full      ewma  ...                 NaN                     NaN

[5 rows x 11 columns]
\n$ head -n 6 $REPORT/full/dm.csv
portfolio,baseline,dm_stat,p_value,n_effective,dm_stat_qlike,p_value_qlike,n_effective_qlike,comparison_valid,comparison_valid_qlike
ew,baseline,,,0,,,0,0,0
ew,lw,,,0,,,0,0,0
ew,oas,,,0,,,0,0,0
mv,baseline,,,0,,,0,0,0
mv,lw,,,0,,,0,0,0
\n$ head -n 10 $REPORT/skip_stats.csv
regime,portfolio,estimator,skip_reason,skip_count,windows,skip_share
calm,ew,baseline,,0,33,0.0
calm,ew,cc,,0,33,0.0
calm,ew,ewma,,0,33,0.0
calm,ew,lw,,0,33,0.0
calm,ew,oas,,0,33,0.0
calm,ew,overlay,,0,33,0.0
calm,ew,quest,,0,33,0.0
calm,ew,rie,,0,33,0.0
calm,ew,sample,,0,33,0.0
\n$ python3 -c "import json;print(json.dumps(json.load(open(\"reports/eval-ticket-15-smoke-aligned5/run.json\")), indent=2) )" (windows block)
{
  "windows": {
    "cap_active": true,
    "cap_sources": [
      "max_windows",
      "window_coverage"
    ],
    "window_coverage": 0.013433637829124127,
    "windows_after_caps": 50,
    "windows_evaluated": 50,
    "windows_requested": 3722
  },
  "cap_active": true,
  "cap_sources": [
    "max_windows",
    "window_coverage"
  ]
}
\n$ overlay rows (full/metrics.csv)
   regime portfolio  ...  comparison_valid_es  comparison_valid_qlike
10   full        ew  ...                  1.0                     1.0
11   full        mv  ...                  1.0                     1.0

[2 rows x 10 columns]
## Summary
- Changes: added per-metric `comparison_valid_*` flags for Δ metrics, DM comparison validity now respects `min_comparison_windows`, run metadata captures `windows_after_caps`; summary sanity aggregation drops capped runs.
- Tests: `. .venv/bin/activate && make test-fast` (pass).
- Smoke: `reports/eval-ticket-15-smoke-aligned5/` (deterministic, capped max_windows=50; delta comparisons valid with n_effective=50; DM n_effective=0; cap_sources=['max_windows','window_coverage']). Not headline—capped/truncated.
- Stop-the-line issues: none observed.
\n## Bundle
Path: docs/gpt_bundles/20251220_080141_ticket-15_20251220_071719_ticket-15_eval-contamination-fixup.zip
$ unzip -l (first entries)
Archive:  docs/gpt_bundles/20251220_080141_ticket-15_20251220_071719_ticket-15_eval-contamination-fixup.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
    32687  2025-12-20 08:01   DIFF.patch
     3587  2025-12-20 08:01   AGENTS.md
        0  2025-12-20 08:01   docs/
     5236  2025-12-20 08:01   docs/DOCS_AND_LOGGING_SYSTEM.md
     7823  2025-12-20 08:01   docs/CODEX_SPRINT_TICKETS.md
        0  2025-12-20 08:01   docs/agent_runs/
        0  2025-12-20 08:01   docs/agent_runs/20251220_071719_ticket-15_eval-contamination-fixup/
