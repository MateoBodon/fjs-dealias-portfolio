# T-012: execute the next daily dow empirical campaign

## Goal

Execute one larger monitored daily `dow` empirical campaign under a fresh `reports/rc-t-012/` root so the next review happens after materially more evidence than T-010. The campaign is a frozen 2×2 matrix inside the accepted empirical-only lane: factor prewhiten `ff5mom` versus `off`, and window `126` versus `252`, with all other scientific settings held fixed. The ticket must finish end-to-end: tracked configs, monitored execution, shared summary pack, campaign-level comparison file, advisor memo, validated run log, and a self-describing GPT bundle.

## Why now

T-011 ratified T-010 without rerunning the heavy jobs and locked the repo onto the daily `dow` empirical-only lane. The next useful review should not be another small patch. It should answer a real empirical question: does the accepted T-008/T-010 behavior survive a longer-window robustness axis, or does the lane become too fragile to scale? This is also the right moment to make long Hetzner monitoring discipline explicit, because both T-008 and T-010 showed how easy it is to misread live runs as blocked while workers are still active.

## Scope

Allowed:

- create and track two additional derived configs:
  - `experiments/eval/config.paper_v1_dow_window252.yaml`
  - `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`
- keep the scientific lane frozen to daily `dow` with:
  - edge mode `tyler`
  - shrinker baseline `rie`
  - horizon `21`
  - assets top `60`
  - deterministic execution
- run exactly four eval legs under `reports/rc-t-012/`:
  - `dow-paper-v1_ff5mom_w126`
  - `dow-paper-v1_noprewhiten_w126`
  - `dow-paper-v1_ff5mom_w252`
  - `dow-paper-v1_noprewhiten_w252`
- run `tools/make_summary.py` on `reports/rc-t-012/`
- create:
  - `reports/rc-t-012/summary/t012_full_regime_comparison.csv`
  - `reports/rc-t-012/summary/campaign_decision.md`
- update:
  - `docs/plan/NOW.md`
  - `project_state/CURRENT_STATE.md`
  - `project_state/KNOWN_ISSUES.md`
  - `PROGRESS.md`
- validate the run log and build the GPT bundle

Excluded:

- no detector, gating, estimator, portfolio, calibration, or data-source code changes
- no daily `week` or weekly `oneway` execution
- no silent overwrite of an existing `reports/rc-t-012/` tree
- no silent rerun of a failed or ambiguous leg inside the same ticket
- no theory-restoring or detector-validating claim
- no hidden reuse of a previous ticket’s run log or bundle metadata

## Acceptance criteria

### A. Frozen matrix and tracked config control

- `experiments/eval/config.paper_v1_dow_window252.yaml` exists and is tracked.
- `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml` exists and is tracked.
- The only intended scientific differences from the accepted T-008/T-010 pair are:
  - window `126` versus `252`
  - factor prewhiten `ff5mom` versus `off`
- The shared daily `dow` lane remains otherwise unchanged:
  - edge mode `tyler`
  - shrinker baseline `rie`
  - horizon `21`
  - assets top `60`
  - seed `0`
  - deterministic execution
- The four run directories above exist under `reports/rc-t-012/`.
- The resolved configs show that all non-window and non-prewhiten settings match across the matrix.

### B. Compute use and monitoring are explicit and auditable

- The ticket records one host-capacity snapshot before the first launch:
  - `uptime`
  - `free -h`
  - `nproc`
  - one `ps -eo pid,etime,pcpu,pmem,cmd` snapshot
- At most two T-012 eval legs are launched concurrently.
- Before launching any second batch, the run log records a fresh host snapshot and explicitly notes whether other non-T-012 load is present.
- Before each launch:
  - the intended output directory does not already exist
  - the run log records the exact command and UTC launch time
- After each launch:
  - the run log records one process snapshot
  - the run log records one initial artifact snapshot from the output directory
- While any T-012 worker remains live:
  - monitoring checks happen at least every 30 minutes
  - each check records:
    - a `ps` snapshot
    - `tail -n 20 run.log`
    - a concise file-surface snapshot for that run directory
- No run is declared blocked while the worker is still live and CPU-active unless the outputs are shown corrupt.
- If a run exits abnormally, looks stalled after the worker exits, or would require overwrite or rerun, the partial tree is preserved and the ticket stops with a documented blocker instead of silently retrying.

### C. Required run and summary surfaces

Each run directory contains, at minimum:

- `run.json`
- `resolved_config.json`
- `full/metrics.csv`
- `full/diagnostics.csv`
- `full/risk.csv`
- `overlay_toggle.md`

The shared summary directory contains, at minimum:

- `reports/rc-t-012/summary/summary_detection.csv`
- `reports/rc-t-012/summary/summary_perf.csv`
- `reports/rc-t-012/summary/summary_skip_stats.csv`
- `reports/rc-t-012/summary/overlay_forensics.csv`
- `reports/rc-t-012/summary/kill_criteria.json`
- `reports/rc-t-012/summary/completeness.json`
- `reports/rc-t-012/summary/t012_full_regime_comparison.csv`
- `reports/rc-t-012/summary/campaign_decision.md`

### D. Honest empirical-only campaign classification

- `t012_full_regime_comparison.csv` contains full-regime rows for all four runs and both portfolios (`ew`, `mv`).
- For each run and portfolio, the comparison file records at least:
  - `delta_qlike_vs_baseline`
  - `delta_mse_vs_baseline`
  - `cap_active`
  - `window_coverage`
  - all `comparison_valid_*` fields
  - `n_effective_qlike`
  - `n_changed`
  - `changed_frac`
- `campaign_decision.md` ends with exactly one explicit classification:
  - `empirical-lane-still-worth-scaling`
  - or `empirical-lane-too-fragile-to-scale`
- The memo explicitly states:
  - whether the two `w126` control legs reproduce the ratified T-010 truth
  - whether either `w252` leg remains uncapped, comparison-valid, and QLIKE-improving for both `ew` and `mv`
  - that daily `dow` is still not detector-validated
  - that daily `dow` remains an empirical-only lane
- If either `w126` control leg fails to reproduce the ratified T-010 empirical gate, or if both `w252` legs fail that gate, the classification must be `empirical-lane-too-fragile-to-scale`.
- If both `w126` controls remain comparison-valid and at least one `w252` leg also remains uncapped, comparison-valid, and QLIKE-improving for both portfolios, the classification may be `empirical-lane-still-worth-scaling`.

### E. Self-describing run log and bundle completeness

- `docs/plan/NOW.md` names T-012 as the active ticket while the campaign is in progress and reflects the final outcome honestly after completion.
- `project_state/CURRENT_STATE.md` and `project_state/KNOWN_ISSUES.md` reflect the campaign result without overclaiming.
- `PROGRESS.md` records the T-012 run and artifact path.
- The T-012 run log exists under `reports/_runs/<run_name>/` and is itself included in the GPT bundle.
- The GPT bundle is self-describing:
  - `BUNDLE_INDEX.md` names `ticket_id: T-012`
  - `BUNDLE_INDEX.md` points `run_dir` at the T-012 run log
- The run log validates successfully.
- The GPT bundle contains:
  - `docs/recenter/POST_ONEWAY_PATH_DECISION.md`
  - `docs/recenter/DAILY_DOW_HEADLINE_CONTRACT.md`
  - this T-012 ticket
  - the T-012 run log
  - the ratified T-010 advisor memo
  - `reports/rc-t-012/summary/campaign_decision.md`
  - `reports/rc-t-012/summary/t012_full_regime_comparison.csv`
  - the core T-012 summary outputs
  - the core outputs from all four T-012 run directories

## Required tests

- `. .venv/bin/activate && make test-fast`
- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py \
  --config experiments/eval/config.paper_v1.yaml \
  --returns-csv data/returns_daily.csv \
  --factors-csv data/factors/ff5mom_daily.csv \
  --out ${RC_ROOT}/dow-paper-v1_ff5mom_w126 \
  --exec-mode deterministic'`
- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py \
  --config experiments/eval/config.paper_v1_dow_noprewhiten.yaml \
  --returns-csv data/returns_daily.csv \
  --factors-csv data/factors/ff5mom_daily.csv \
  --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w126 \
  --exec-mode deterministic'`
- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py \
  --config experiments/eval/config.paper_v1_dow_window252.yaml \
  --returns-csv data/returns_daily.csv \
  --factors-csv data/factors/ff5mom_daily.csv \
  --out ${RC_ROOT}/dow-paper-v1_ff5mom_w252 \
  --exec-mode deterministic'`
- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py \
  --config experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml \
  --returns-csv data/returns_daily.csv \
  --factors-csv data/factors/ff5mom_daily.csv \
  --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w252 \
  --exec-mode deterministic'`
- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python tools/make_summary.py --rc-dir ${RC_ROOT}'`
- `python - <<'PY'`
  artifact and comparison assertion covering:
  all four run directories, required full-output surfaces, required summary
  surfaces, both portfolios in all four runs, mandatory baseline rows in all
  four full metrics surfaces, `t012_full_regime_comparison.csv`, and
  `campaign_decision.md`
  `PY`
- `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/<run_name>`
- `python - <<'PY'`
  bundle-contents assertion covering:
  the daily `dow` decision docs, the T-012 ticket, the T-012 run log, the
  ratified T-010 advisor memo, `campaign_decision.md`,
  `t012_full_regime_comparison.csv`, and core outputs from all four T-012 runs
  `PY`

## Required artifacts

- `experiments/eval/config.paper_v1_dow_window252.yaml`
- `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`
- `reports/rc-t-012/`
- run log in `reports/_runs/`
- GPT bundle in `reports/_bundles/`
- updated `PROGRESS.md`
- updated `docs/plan/NOW.md`
- updated `project_state/CURRENT_STATE.md` if repo truth changes materially
- updated `project_state/KNOWN_ISSUES.md` if repo truth changes materially

## Risks / invariants

- keep the claim boundary empirical-only throughout
- do not silently retry or overwrite a partial long-run tree
- do not reinterpret T-010’s ratified empirical result as detector validation
- do not widen beyond the frozen `window × prewhiten` matrix
- do not hide weak `w252` behavior behind within-run baseline changes or incomplete monitoring
- do not let the bundle reuse prior-ticket metadata again; the review pack must identify itself as T-012

## Stop conditions

- stop if `reports/rc-t-012/` already exists
- stop if either `window252` config cannot be expressed as a window-only/prewhiten-only derivation of the accepted daily `dow` pair
- stop if any run requires overwrite or silent rerun
- stop if any run output is shown corrupt
- stop if summary generation fails or the run log cannot be validated
- stop if continuing would require detector, gating, estimator, portfolio, calibration, or data-source changes instead of this bounded empirical matrix
- stop if measured host contention from other active jobs is high enough that launching or continuing the planned batch would be irresponsible; record the measured contention and stop rather than guessing