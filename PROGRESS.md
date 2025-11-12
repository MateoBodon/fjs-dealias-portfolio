## 2025-11-12T10:10Z — prewhiten RC-lite + coverage lift (feat/prewhiten-coverage@baa1a4b)
- **Data**: `data/returns_daily.csv` (`sha256=96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) + `data/factors/ff5mom_daily.csv` (`sha256=469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), revalidated via `tools/verify_dataset.py` before every AWS dispatch.
- **Commands**:
  1. `make test-fast`
  2. `make aws:rc-lite EXEC_MODE=deterministic`
  3. `INSTANCE_DNS=… KEY_PATH=… RC_REQUIRE_ISOLATED=0 RC_DOW_MIN_REPS=10 make aws:rc-dow`
  4. `INSTANCE_DNS=… KEY_PATH=… RC_VOL_MIN_REPS∈{8,6,4,2} RC_GATE_DELTA_FRAC_MIN∈{0.01,0.007,0.005} RC_OVERLAY_DELTA∈{0.05,0.04,0.03} make aws:rc-vol`
  5. `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251112`
  6. `make memo` and `PYTHONPATH=src:. python tools/build_brief.py --config experiments/equity_panel/config.rc.yaml`
- **Artifacts**:
  - `reports/rc-20251112/{dow-tyler,vol-tyler}/` (full/calm/crisis CSVs, diagnostics detail, `prewhiten_diagnostics.csv`, plots, `resolved_config.json`).
  - `reports/rc-20251112/run_manifest.json` (datasets + AWS run IDs `20251112T035651Z` dow, `20251112T084450Z` vol) and `metrics_summary.json`.
  - Refreshed `reports/memo.md`, `reports/memo_20251112_092343.md`, `reports/brief.md`, `reports/brief_20251112_092348.md`, plus `figures/rc/**`.
- **Coverage / deltas**:
  - DoW (tyler edge, soft gate, q≤2) now lands at detection_rate full ≈ 3.73 %, calm ≈ 4.13 %, crisis ≈ 3.83 % with ΔMSE (overlay vs baseline) ≲3e‑10 and DM stats still undefined because almost every window flips.
  - Vol-state (tyler edge, soft gate, q≤2) is trending upward but still shy of the 2–6 % target: detection_rate full ≈ 0.43 %, calm ≈ 0.14 %, crisis ≈ 0.70 %; `percent_changed` ~10.7 %. Relaxing `group_min_replicates`/`min_reps_vol` down to 2 and lowering `gate_delta_frac_min` to 0.5 % improved acceptance, but the memo + README track the remaining gap as an open item.
- **Notes**:
  - Prewhitening CLI/telemetry is fully wired (CLI flags, `prewhiten_summary.json`, `prewhiten_diagnostics.csv` per window) and covered by `tests/test_equity_prewhiten.py`.
  - Makefile exposes `RC_DOW_MIN_REPS`, `RC_VOL_MIN_REPS`, `RC_VOL_GROUP_REPS`, and `RC_REQUIRE_ISOLATED` so RC-lite targets stay configurable without editing YAML.
  - README Current Status references `reports/rc-20251112/` and punts AWS host specifics to `docs/CLOUD.md` per ops request.
  - Calibration defaults remain untouched; any future tuning still needs the “before/after” note per AGENTS.md.

## 2025-11-05T21:52Z — sprint-1 calibration & smoke (2d235955)
- **Data**: `data/returns_daily.csv` (`sha1=1ff062eab6f0741f7fdc8d25098ffb8f9e3a5344`)
- **Commands**: `make sweep:acceptance`, `make run:equity_smoke`, `make memo`, `make test`
- **Artifacts**: `reports/figures/roc_null.png`, `reports/figures/roc_power.png`, `calibration_defaults.json`, `reports/synthetic/{null_harness,power_harness}/`, `experiments/equity_panel/outputs_smoke/`, `reports/memo.md`
- **Notes**: Synthetic harness writes score tables + calibration defaults (energy floor), detection summary now logs edge bands/gating mode and MV solver stats, MV defaults locked to ridge=1e-4 with box [0,0.1] and 5 bps turnover cost cap.

## 2025-11-07T01:30Z — deterministic DoW + vol RC on WRDS (a03a3764)
- **Data**: `data/returns_daily.csv` (sha256=`96ac7dd3…3197`, verified against `data/registry.json` before every run via `tools/verify_dataset.py`).
- **Commands**: `make test-fast`, `make aws:test-fast MODE=deterministic`, `make aws:rc-dow AWS_ARGS="EDGE=tyler MODE=deterministic"`, `make aws:rc-dow AWS_ARGS="EDGE=scm MODE=deterministic"`, `make aws:rc-vol AWS_ARGS="EDGE=tyler MODE=deterministic"`. (Each rc target uses deterministic BLAS/thread caps via `scripts/aws_run.sh`.)
- **Artifacts**: `reports/rc-20251107/{dow-tyler,dow-scm,vol-tyler}/`, histograms (`acceptance_hist_*.png`, `edge_margin_hist_*.png`), extended DM tables (`dm.csv` now includes LW/OAS contrasts), `reports/rc-20251107/summary_stats.json`, `reports/rc-20251107/run_manifest.json`, calibration log (`reports/calibration_notes.md`), updated `reports/memo.md`.
- **Notes**: All regimes still report 0% acceptance because `detect_spikes` throws `detection_error`; we dropped δ by 0.05 in `calibration/defaults.json` per the crisis decision rule and reran, but failures persisted (documented in `reports/calibration_notes.md`). MV defaults (ridge 1e-4, box [0,0.1], 5 bps turnover, κ-cap 1e6) are enforced everywhere and windows breaching the condition cap are skipped. QLIKE DM tests against LW/OAS remain finite (e.g., DoW full: EW p≈0.003, MV p≈0.009), so the memo now lists those contrasts even though MSE DM stats are undefined. Representative AWS run IDs: `20251107T012713Z` (DoW–Tyler, 45 s), `20251107T012811Z` (DoW–SCM, 45 s), `20251107T012910Z` (Vol–Tyler, 40 s); all completed with status 0 and synced back to `reports/aws/`.

## 2025-11-07T19:25Z — deterministic RC rerun w/ FF5+MOM registry (b1611887)
- **Data**: `data/returns_daily.csv` (sha256 `96ac7dd3…3197`) + `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), both checked via `tools/verify_dataset.py` / `data/factors/registry.json`.
- **Commands**: `make test-fast`; `MODE=deterministic make aws:test-fast` *(blocked: missing `INSTANCE_DNS`)*; `USE_FACTORS=1 EDGE=tyler make rc-dow`; `USE_FACTORS=1 EDGE=scm make rc-dow`; `USE_FACTORS=1 EDGE=tyler make rc-vol`; `EDGE=tyler MODE=deterministic USE_FACTORS=1 make aws:rc-dow` *(blocked: missing `INSTANCE_DNS`)*; `EDGE=tyler MODE=deterministic USE_FACTORS=1 make aws:rc-vol` *(blocked: missing `INSTANCE_DNS`)*.
- **Artifacts**: Refreshed `reports/rc-20251107/{dow-tyler,dow-scm,vol-tyler}/` (each with `run.json`, `resolved_config.json`, updated `diagnostics*.csv`, `dm.csv`, and the new `acceptance_hist_{dow,vol}.png` / `edge_margin_hist_{dow,vol}.png`). Histogram + percent-changed diagnostics feed into the new “Transfer Check” memo section.
- **Notes**: Factor prewhitening now lands on `ff5mom` with `prewhiten_r2_mean ≈ 0.39` and `factor_present_share = 1`. Despite that, overlay gating never fired (`percent_changed = 0` across all regimes; DM `n_effective = 0` and ΔMSE = 0), and reason codes remain dominated by `detection_error` (DoW) and `balance_failure` (vol). Baseline κ̄ stayed mild for DoW (≈11) but vol crisis κ̄ ≈79, highlighting how unbalanced the vol grouping still is. AWS reruns remain blocked until the required SSH environment variables are populated; once available we should re-dispatch `aws:test-fast`, `aws:rc-dow`, and `aws:rc-vol` to capture the same factor-registry outputs on the EC2 runner.

## 2025-11-09T09:10Z — AWS detector telemetry + sensitivity sweep (245fa52)
- **Data**: `data/returns_daily.csv` (sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) and `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), both revalidated via `tools/verify_dataset.py` / `data/registry.json` and the mirrored alias we added to `data/factors/registry.json`.
- **Commands**: `make test-fast`; `INSTANCE_DNS=ec2-98-92-104-129.compute-1.amazonaws.com KEY_PATH=~/.ssh/mateo-us-east-1-ec2-2025 AWS_ARGS="MODE=deterministic" make aws:test-fast`; local RCs for `USE_FACTORS=1 EDGE∈{tyler,scm}` across `rc-dow`, `rc-vol`, `rc-week`, `rc-dowxvol`; deterministic AWS mirrors via `make aws:rc-dow` (tyler + scm), `make aws:rc-vol`, `make aws:rc-week`, `make aws:rc-dowxvol` with `USE_FACTORS=1 EDGE=tyler MODE=deterministic`; `RC_SENS_START=2024-03-01 RC_SENS_END=2024-10-31 make rc-sensitivity`; `make inject-spike` (now routed through `RC_PY`).
- **Artifacts**: 
  - AWS run manifests + telemetry: `reports/aws/20251109T015457Z/runs/20251109T015457Z/run.json` (test-fast), `reports/aws/20251109T015712Z/runs/20251109T015712Z/run.json`, `.../015833Z/...`, `.../020327Z/...`, `.../085041Z/...`, `.../085125Z/...` for the five RC targets.
  - AWS RC outputs: `reports/aws/20251109T015712Z/rc-20251109/dow-tyler/`, `reports/aws/20251109T015833Z/rc-20251109/dow-scm/`, `reports/aws/20251109T020327Z/rc-20251109/vol-tyler/`, `reports/aws/20251109T085041Z/rc-20251109/week/`, `reports/aws/20251109T085125Z/rc-20251109/dowxvol/` (each with `run.json`, `diagnostics.csv`, `diagnostics_detail.csv`, per-regime histograms such as `design_ok_full_hist.png`).
  - Sensitivity sweep: `reports/rc-sensitivity/rc-sensitivity-20251108/{run_manifest.json,tables/sensitivity_summary.csv,tables/changed_windows.csv,figures/acceptance_rate_ri[0|1]_align[0p70|0p80|0p90].png}`.
  - Weak-spike study: `reports/figures/inject_summary.csv`, `reports/figures/{inject_recall.png,inject_fp.png,inject_manifest.json}`.
- **Notes**: All deterministic RCs (including the new week/dow×vol designs) remain at zero acceptance: the enriched telemetry shows `gating_initial = raw_detection_count = 0` everywhere, DoW design compliance tops out at ~59%, vol-state at ~11%, and week slices at ~32%, so reason codes split between `detection_error` and `balance_failure` (see `.../diagnostics_detail.csv`). The sensitivity sweep over 72 `(require_isolated, alignment_min_cos, delta_frac, stability_eta)` combos reports `n_changed_windows = 0` for every cell, so η/δ/alignment tweaks alone cannot unlock detections while the balance issues persist. The weak-spike harness injected μ∈{3,4,5} into ~8% of 1,446 windows yet still logged 0% recall and 0% FP, underscoring that the detector is never emitting candidates on the current balanced panels.

## 2025-11-10T02:08Z — Plan for prewhiten/coverage refresh (284034b1)
- **Scope**: Wire prewhitening + factor diagnostics, relax nested guardrails, rerun RC-lite + ROC sweeps per operator brief.
- **Plan**:
  1. Inspect runners/configs for current prewhiten + diagnostics plumbing; outline manifest schema changes.
  2. Add `--prewhiten/--factor-csv` to eval + panel runners, persist R² + factor names into diagnostics + `run_manifest.json`, update README/memo templates, and cover with unit/integration tests.
  3. Loosen nested balancing constraints, surface `--gate-delta-frac-min`/`q_max` configs, and validate via nested smoke run to ensure skip reasons diversify.
  4. Dispatch deterministic `aws:rc-lite` against WRDS registry data, sync artifacts + telemetry back, and log in PROGRESS.
  5. Execute acceptance ROC sweep (`aws:calibrate-thresholds`), consolidate artifacts + update `calibration_defaults.json`, refresh configs, and prep PR.

## 2025-11-10T03:35Z — Prewhiten plumbing + RC-lite/AWS calibration status (284034b1)
- **Data**: `data/returns_daily.csv` (sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) and `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`), verified via `tools/verify_dataset.py` before each local/AWS invocation.
- **Commands**: `pytest tests/experiments/test_prewhiten_utils.py`; `pytest tests/test_equity_prewhiten.py -m slow`; `make test-fast`; `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --prewhiten ff5mom --factor-csv data/factors/ff5mom_daily.csv --no-progress --exec-mode deterministic` (local nested smoke to validate relaxed balancing + per-window factor telemetry); `INSTANCE_DNS=ec2-98-92-104-129.compute-1.amazonaws.com KEY_PATH=~/.ssh/mateo-us-east-1-ec2-2025 AWS_ARGS="EXEC_MODE=deterministic" make aws:rc-lite`; `INSTANCE_DNS=... AWS_ARGS="EXEC_MODE=deterministic CALIB_TRIALS_NULL=600 CALIB_TRIALS_ALT=600" make aws:calibrate-thresholds` (in flight; monitor shows `calibration_progress: 1/48` at run id `20251110T031745Z`).
- **Artifacts**:
  - Local nested smoke: `experiments/equity_panel/outputs_nested_smoke/nested_J5_solver-auto_est-dealias_prep-prewhiten_factorsMKT,SMB,HML,RMW,CMA,MOM-prewhiten_modeff5mom/{rolling_results.csv,summary.json,run_meta.json}` now carry `prewhiten_*` columns plus relaxed nested skip detail (`weeks_common`, `years_dropped`, `replicates_used`).
  - RC-lite AWS batch (run dir `reports/aws/20251110T025119Z/`): refreshed smoke/crisis outputs with memo+brief regenerated (`reports/aws/20251110T025119Z/memo_20251110_025534.md`, `.../brief.md`) and gallery snapshots (`figures/rc/**`). Each run folder contains the augmented diagnostics columns (`prewhiten_*`, `factor_present`).
  - AWS calibration sweep is still running under `reports/aws/runs/20251110T031745Z` (not yet synced locally); progress + logs available via `reports/runs/monitor` once the EC2 job finishes. No new `calibration/edge_delta_thresholds.json` committed yet—pending that run’s completion.
- **Notes**: Evaluation + panel runners accept `--prewhiten`/`--factor-csv` and persist telemetry into diagnostics, summary payloads, and `run.json`. Memo/brief templates now expose a “Factor Baseline” block. Nested guardrails accept looser replicate/ISO-week intersections (records `replicates_used`, `years_dropped`) and daily runners surface `RC_Q_MAX`/`RC_GATE_DELTA_FRAC_MIN`. `make test-fast` + the new targeted tests cover the refactor. RC-lite on AWS completed deterministically; calibration sweep remains queued (ETA ≈ 9h per `run_monitor`), so defaults will be updated once that run lands.

## 2025-11-11T01:45Z — Deterministic AWS calibration sweep (20251110T154048Z)
- **Data**: Same WRDS returns + FF5+MOM factors as prior RC runs (hashes above), re-verified before dispatch.
- **Commands**: `INSTANCE_DNS=ec2-98-92-104-129.compute-1.amazonaws.com KEY_PATH=~/.ssh/mateo-us-east-1-ec2-2025 AWS_ARGS="EXEC_MODE=deterministic CALIB_TRIALS_NULL=600 CALIB_TRIALS_ALT=600" make aws:calibrate-thresholds`; post-run rsync of `reports/runs/20251110T154048Z/` and `calibration/*`; `make test-fast`.
- **Artifacts**:
  - Remote provenance: `reports/aws/20251110T154048Z/runs/20251110T154048Z/run.json` (status `0`, duration ≈10 h) plus full `metrics.jsonl`/`progress.jsonl`.
  - Updated calibration files in repo: `calibration/edge_delta_thresholds.json` (48 cells) + `calibration/defaults.json` (new selection + metadata).
  - Local monitor log `aws_calib_latest.log` capturing `[1/48 … 48/48]` milestones.
- **Notes**: Sweep covered `(p ∈ {64, 80, 96}, replicates ∈ {14, 20}, δ_abs ∈ {0.35, 0.45, 0.55, 0.65}, edge ∈ {scm, tyler})` under deterministic thread caps. Final ETA ticked to zero at 48/48 with no retries. With new thresholds committed, RC configs should continue pointing at `calibration/edge_delta_thresholds.json` (rev 20251110). Unit tests re-run locally (`make test-fast`) to confirm no regressions after syncing the calibration outputs.

## 2025-11-11T02:00Z — RC-lite refresh w/ new calibration defaults (284034b1)
- **Data**: `data/returns_daily.csv` + `data/factors/ff5mom_daily.csv` (hashes above), verified pre-run.
- **Commands**: `make rc-lite` (covers smoke + 2020 crisis configs across {dealias,lw,oas}); gallery/memo/brief regenerated implicitly by the target.
- **Artifacts**:
  - Smoke outputs: `experiments/equity_panel/outputs_smoke/oneway_J5_solver-auto_est-{dealias,lw,oas}_prep-*`.
  - Crisis outputs: `experiments/equity_panel/outputs_crisis_2020/...`.
  - Gallery: `figures/rc/**` (new tables/plots incorporate `prewhiten_*` columns).
  - Memo/brief refreshed under `reports/{memo.md,brief.md}` plus timestamped copies.
- **Notes**: First full rc-lite after ingesting the 20251110 calibration defaults. No runtime errors; diagnostics now show factor baselines + updated Δ thresholds. Ready to mirror on AWS (`make aws:rc-lite`) if we want cloud telemetry.
