# Results

- Initialized `reports/_runs/20260331_230430_t-012` because no existing T-012
  run log was present under `reports/_runs/`.
- The helper path referenced by the `runlog-init` skill does not exist in this
  repo (`tools/agentic/runlog_init.py` missing), so the canonical checked-in
  fallback `make -f Makefile.agentic init-runlog TICKET=T-012` was used
  successfully instead.
- Created the two required config-only T-012 derivations:
  - `experiments/eval/config.paper_v1_dow_window252.yaml`
  - `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`
- Verified the intended config-only differences before launch:
  - base `ff5mom w126` vs `ff5mom w252`: `window` only
  - no-prewhiten `w126` vs no-prewhiten `w252`: `window` only
  - `ff5mom w252` vs `noprewhiten w252`: `prewhiten` and
    `use_factor_prewhiten` only
- Required fast tests passed before any eval launch.
- Prelaunch host snapshot at `2026-03-31 23:06:29 UTC`:
  - `uptime`: `load average: 0.62, 0.25, 0.09`
  - `free -h`: `125Gi` total RAM, `119Gi` available, swap unused
  - `nproc`: `96`
  - `ps` shows only light editor / Codex background load and no visible
    non-T-012 `experiments/eval/run.py` worker
- Verified both first-batch output paths were absent before launch:
  - `reports/rc-t-012/dow-paper-v1_ff5mom_w126`
  - `reports/rc-t-012/dow-paper-v1_noprewhiten_w126`
- Launch times recorded:
  - `ff5mom_w126`: `2026-03-31 23:06:42 UTC`
  - `noprewhiten_w126`: `2026-03-31 23:06:43 UTC`
- Launched the first monitored batch under `reports/rc-t-012/`:
  - `dow-paper-v1_ff5mom_w126`
  - `dow-paper-v1_noprewhiten_w126`
- Immediate post-launch process snapshot at `2026-03-31 23:06:56 UTC` showed
  both workers live and CPU-active:
  - `pid=164412` for `dow-paper-v1_ff5mom_w126`
  - `pid=164416` for `dow-paper-v1_noprewhiten_w126`
- Immediate post-launch file snapshots showed the expected init-stage artifacts
  for both runs:
  `prewhiten_diagnostics.csv`, `prewhiten_summary.json`,
  `resolved_config.json`, `run.json`, and `run.log`.
- Early live-monitor re-check while both workers remained active:
  - `ps` still showed both T-012 workers CPU-active after about `01:10`
    elapsed
  - both `run.log` tails still showed only the initial
    `START ... stage=init` line
- `run.json` then moved to `stage=evaluate` / `status=running` for both live
  control legs while keeping the expected frozen settings:
  - `dow-paper-v1_ff5mom_w126`: `config_path=experiments/eval/config.paper_v1.yaml`
  - `dow-paper-v1_noprewhiten_w126`: `config_path=experiments/eval/config.paper_v1_dow_noprewhiten.yaml`
- The two required `window252` configs are now staged in git, so the tracked
  config-control surface is already satisfied before the heavy runs finish.
- Monitoring snapshot at `2026-03-31 23:20:12 UTC`:
  - both T-012 workers still live and CPU-active after about `13:23` elapsed
  - `run.log` still contains only the initial `START ... stage=init` line for
    both runs
  - the on-disk file surface remains the same five init-stage files in each
    directory
- Monitoring snapshot at `2026-03-31 23:40:56 UTC`:
  - both T-012 workers still live and CPU-active after about `34:07` elapsed
  - `run.log` remains unchanged for both runs
  - the file surface still shows only
    `prewhiten_diagnostics.csv`, `prewhiten_summary.json`,
    `resolved_config.json`, `run.json`, and `run.log`
  - `run.json` still reports `stage=evaluate` / `status=running` for both
    runs, so this remains the same observability-limited but not-yet-blocked
    state seen in T-008/T-010 rather than evidence of corruption
- Confirmed from `tests/test_validate_runlog.py` that the validator accepts an
  in-progress run log, so partial validation is legitimate for this checkpoint.
- The current T-012 run log validates successfully:
  `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
  returned `"valid": true` with no errors or warnings.
- Built an interim T-012 review bundle at
  `reports/_bundles/20260331_234217_T-012_gpt_bundle.zip`.
- That first interim bundle exposed one ticket-relevant packaging gap:
  `scripts/make_gpt_bundle.py` did not yet include the two new `window252`
  config files, so the bundle was not fully self-describing.
- Patched `scripts/make_gpt_bundle.py` minimally to include:
  - `experiments/eval/config.paper_v1_dow_window252.yaml`
  - `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`
- Rebuilt the interim bundle at
  `reports/_bundles/20260331_234249_T-012_gpt_bundle.zip`.
- Verified the rebuilt interim bundle contains the key currently available
  T-012 surfaces:
  the daily `dow` decision docs, the T-012 ticket, both new configs, the T-012
  run log, and the partial `w126` run directories under `reports/rc-t-012/`.
- Ran the narrow bundle regression test after patching the helper:
  `. .venv/bin/activate && pytest -q tests/test_gpt_bundle.py`
  passed with `2 passed`.
- At `2026-04-01 00:04:35 UTC`, the first launched `w126` batch was no longer
  live:
  no `experiments/eval/run.py` worker remained, but both run directories still
  reported `status=running` / `stage=evaluate`, `run.log` remained stuck on the
  initial `START ... stage=init` line, and only the init-stage files plus empty
  `calm/`, `crisis/`, and `full/` directories were present.
- The user then explicitly confirmed an accidental interrupt and authorized a
  restart.
- Preserved the interrupted partial trees without overwrite under:
  `reports/rc-t-012/_preserved_interrupted_20260401_000435/`
- Verified the required launch paths were absent again after preservation.
- Recorded a fresh restart host snapshot at `2026-04-01 00:04:51 UTC`:
  - `uptime`: `load average: 0.16, 0.96, 1.56`
  - `free -h`: `125Gi` total RAM, `119Gi` available, swap unused
  - `nproc`: `96`
  - `ps` again showed only light editor / Codex background load and no visible
    non-T-012 `experiments/eval/run.py` worker
- Relaunched the `w126` control batch at `2026-04-01 00:04:58 UTC`:
  - `dow-paper-v1_ff5mom_w126`
  - `dow-paper-v1_noprewhiten_w126`
- Immediate post-relaunch process snapshot at `2026-04-01 00:05:06 UTC` showed
  both workers live and CPU-active:
  - `pid=180956` for `dow-paper-v1_ff5mom_w126`
  - `pid=180959` for `dow-paper-v1_noprewhiten_w126`
- Immediate post-relaunch file snapshots again show the expected init-stage
  artifacts for both runs:
  `prewhiten_diagnostics.csv`, `prewhiten_summary.json`,
  `resolved_config.json`, `run.json`, and `run.log`.
- Monitoring snapshot at `2026-04-01 00:15:56 UTC`:
  - both relaunched workers still live and CPU-active after about `10:58`
    elapsed
  - both `run.log` tails still showed only the initial
    `START ... stage=init` line
  - both `run.json` files still reported `stage=evaluate` / `status=running`
  - the on-disk file surface remained the same five init-stage files in each
    run directory
- Monitoring snapshot at `2026-04-01 00:26:13 UTC`:
  - both relaunched workers still live and CPU-active after about `21:14`
    elapsed
  - `run.log` remained unchanged for both runs
  - the file surface still showed only the five init-stage artifacts in each
    run directory
- Monitoring snapshot at `2026-04-01 00:36:29 UTC`:
  - both relaunched workers still live and CPU-active after about `31:30`
    elapsed
  - `run.log` still remained unchanged for both runs
  - the file surface still showed only the five init-stage artifacts in each
    run directory
  - this satisfies the ticket’s requirement to keep explicit monitoring checks
    within 30 minutes while a worker remains live
- Later monitoring checkpoints confirmed the same live-but-observability-poor
  state at:
  - `2026-04-01 00:47:06 UTC` (`42:08` elapsed)
  - `2026-04-01 00:57:22 UTC` (`52:23` elapsed)
  - `2026-04-01 01:07:41 UTC` (`01:02:43` elapsed)
  - `2026-04-01 01:18:04 UTC` (`01:13:05` elapsed)
  - `2026-04-01 01:28:22 UTC` (`01:23:24` elapsed)
  - `2026-04-01 01:38:56 UTC` (`01:33:58` elapsed)
  - `2026-04-01 01:49:24 UTC` (`01:44:26` elapsed)
  - `2026-04-01 01:59:50 UTC` (`01:54:51` elapsed)
  - `2026-04-01 02:10:17 UTC` (`02:05:18` elapsed)
  - `2026-04-01 02:20:42 UTC` (`02:15:43` elapsed)
  - `2026-04-01 02:31:06 UTC` (`02:26:08` elapsed)
  - `2026-04-01 02:41:29 UTC` (`02:36:31` elapsed)
  - `2026-04-01 02:51:50 UTC` (`02:46:51` elapsed)
  - `2026-04-01 03:02:18 UTC` (`02:57:19` elapsed)
- Despite the nearly static log/file surface during those windows, both workers
  eventually completed cleanly without rerun:
  - `dow-paper-v1_ff5mom_w126` ended at
    `2026-04-01T03:03:37.638920+00:00`
  - `dow-paper-v1_noprewhiten_w126` ended at
    `2026-04-01T03:06:41.892321+00:00`
- Final `w126` completion state:
  - no live `experiments/eval/run.py` worker remains for `reports/rc-t-012/`
  - both `run.json` files report `status=ok` and `stage=complete`
  - both run directories now contain the required full-output surfaces,
    including `full/metrics.csv`, `full/diagnostics.csv`, `full/risk.csv`, and
    `overlay_toggle.md`
- Fresh host snapshot before launching the second batch at
  `2026-04-01 03:07:13 UTC`:
  - `uptime`: `load average: 0.58, 1.39, 1.77`
  - `free -h`: `125Gi` total RAM, `119Gi` available, swap unused
  - `nproc`: `96`
  - `ps` showed only light background editor / Codex load and no live T-012
    worker
- Verified both `w252` output paths were absent before launch:
  - `reports/rc-t-012/dow-paper-v1_ff5mom_w252`
  - `reports/rc-t-012/dow-paper-v1_noprewhiten_w252`
- Launched the second monitored batch at `2026-04-01 03:07:24 UTC`:
  - `dow-paper-v1_ff5mom_w252`
  - `dow-paper-v1_noprewhiten_w252`
- Immediate post-launch process snapshot at `2026-04-01 03:07:38 UTC` showed
  both `w252` workers live and CPU-active:
  - `pid=213251` for `dow-paper-v1_ff5mom_w252`
  - `pid=213258` for `dow-paper-v1_noprewhiten_w252`
- Immediate post-launch file snapshots for both `w252` runs show the expected
  init-stage artifacts:
  `prewhiten_diagnostics.csv`, `prewhiten_summary.json`,
  `resolved_config.json`, `run.json`, and `run.log`.
- Later `w252` monitoring checkpoints confirmed the same live-but-observability-poor
  state at:
  - `2026-04-01 03:52:02 UTC` (`44:38` elapsed)
  - `2026-04-01 04:02:21 UTC` (`54:57` elapsed)
  - `2026-04-01 04:12:39 UTC` (`01:05:15` elapsed)
  - `2026-04-01 04:22:59 UTC` (`01:15:34` elapsed)
  - `2026-04-01 04:33:40 UTC` (`01:26:16` elapsed)
  - `2026-04-01 04:44:13 UTC` (`01:36:49` elapsed)
  - `2026-04-01 04:54:42 UTC` (`01:47:18` elapsed)
  - `2026-04-01 05:05:07 UTC` (`01:57:43` elapsed)
  - `2026-04-01 05:15:41 UTC` (`02:08:17` elapsed)
  - `2026-04-01 05:26:06 UTC` (`02:18:42` elapsed)
  - `2026-04-01 05:36:28 UTC` (`02:29:04` elapsed)
- Across those checkpoints, both `w252` workers remained CPU-active while
  `run.log` still showed only the initial `START ... stage=init` line and the
  on-disk file surface remained limited to the five init-stage artifacts; this
  repeated the same observability weakness already seen in T-008/T-010 and did
  not constitute corruption while the workers were still live.
- `dow-paper-v1_ff5mom_w252` then completed cleanly at
  `2026-04-01T05:45:32.547105+00:00`.
- `dow-paper-v1_noprewhiten_w252` then completed cleanly at
  `2026-04-01T05:48:23.079791+00:00`.
- Final `w252` completion state at `2026-04-01 05:48:40 UTC`:
  - no live `experiments/eval/run.py` worker remains for `reports/rc-t-012/`
  - both `run.json` files report `status=ok` and `stage=complete`
  - both run directories now contain the required full-output surfaces,
    including `full/metrics.csv`, `full/diagnostics.csv`, `full/risk.csv`, and
    `overlay_toggle.md`
- Ran the exact shared summary step under `.venv` for `reports/rc-t-012/`.
- The summary step wrote the required shared summary surfaces:
  - `summary_detection.csv`
  - `summary_perf.csv`
  - `summary_skip_stats.csv`
  - `overlay_forensics.csv`
  - `kill_criteria.json`
  - `limitations.md`
  - `completeness.json`
- Wrote `reports/rc-t-012/summary/t012_full_regime_comparison.csv` with the
  required eight full-regime rows: all four runs and both portfolios.
- Wrote `reports/rc-t-012/summary/campaign_decision.md`.
- Verified the T-012 control reproduction and robustness truth from the
  completed summary surfaces:
  - both `w126` control legs reproduce the ratified T-010 full-regime truth
    exactly for both `ew` and `mv`
  - both `w252` legs remain uncapped, full-coverage, comparison-valid, and
    QLIKE-improving versus baseline for both portfolios
  - the `w252` QLIKE improvements are materially smaller than the corresponding
    `w126` controls, especially in the prewhitened leg
- T-012's explicit campaign classification is
  `empirical-lane-still-worth-scaling`.
- The claim boundary remains explicit:
  - daily `dow` is still not detector-validated
  - daily `dow` remains an empirical-only lane
  - `kill_criteria.json` remains a stricter diagnostic surface than the active
    headline gate
  - `completeness.json` still leaves aggregate coverage-count fields null, but
    that remains a summary-surface limitation rather than missing-run evidence
- The required T-012 artifact/comparison assertion passed over the completed
  four-run tree and the shared summary pack.
- Updated the required state docs to the final T-012 local truth:
  - `docs/plan/NOW.md`
  - `project_state/CURRENT_STATE.md`
  - `project_state/KNOWN_ISSUES.md`
- The completed T-012 run log validates successfully:
  `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
  returned `"valid": true` with no errors or warnings.
- Built a first final-review bundle candidate at
  `reports/_bundles/20260401_055651_T-012_gpt_bundle.zip`.
- That first final bundle exposed one last ticket-relevant packaging gap:
  it still omitted `reports/rc-t-010/summary/advisor_decision.md`, which the
  ticket explicitly requires.
- Patched `scripts/make_gpt_bundle.py` minimally again to include:
  - `reports/rc-t-010/summary/advisor_decision.md`
- Re-ran the narrow bundle regression after that helper patch:
  `. .venv/bin/activate && pytest -q tests/test_gpt_bundle.py`
  passed with `2 passed`.
- Rebuilt the final T-012 review bundle at
  `reports/_bundles/20260401_055651_T-012_gpt_bundle.zip`.
- Verified the rebuilt final bundle contains the required T-012 ticket and run
  log, the key daily `dow` decision docs, the ratified T-010 advisor memo, the
  T-012 summary outputs, and the core outputs from all four completed T-012
  run directories.
- After finalizing `META.json`, `PROGRESS.md`, and the T-012 run-log files,
  re-validated the completed run log again; the validator still returned
  `"valid": true` with no errors or warnings.
- Rebuilt the same final bundle path one last time so it captures the finished
  run-log bookkeeping state.
- Re-ran the bundle assertion against that rebuilt zip, and it still passed.
