## 2025-12-18T22:59Z — Cap contamination transparency (ticket-02-caps-contamination, sha f2b0b8d6)
- **Changes**: Added run_manifest JSONs for eval + equity runs (config/dataset hashes, window counts, cap flags, exec metadata); run.json mirrors manifest. Summary tooling (make_summary, summarize_rc_sanity) now surfaces `windows_total/evaluated`, `max_windows`, `cap_active`, `window_coverage`, and drops capped/incomplete runs from aggregates. Makefile exposes `RC_MAX_WINDOWS` (default off) for rc-lite-sanity daily evals; defaults remain uncapped.
- **Commands**: PATH=.venv/bin:$PATH make test-fast; rc-lite-sanity cap-off (stamp `20251218_223414`) with manual completion of weekly runs; rc-lite-sanity cap-on `RC_MAX_WINDOWS=5` (stamp `20251218_230000`).
- **Artifacts**: Cap-off outputs `reports/rc-20251218-sanity-20251218_223414/` + weekly `experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/{dow-weekly,nested}`; Cap-on outputs `reports/rc-20251218-sanity-20251218_230000/` + weekly `experiments/equity_panel/outputs_rc-lite-20251218_20251218_230000/{dow-weekly,nested}`; summaries + summary_sanity.json for both.
- **Results**: Cap-off daily runs coverage=1 (dow detection≈5.5%, percent_changed=1.0; vol detection≈5.2%, percent_changed≈0.94); weekly runs coverage=0.5 with gating_skip exclusions. Cap-on daily runs flagged cap_active (max_windows=5, window_coverage≈0.09) and excluded; weekly still excluded for gating_skip. Summary files now plainly report cap status and window counts.

## 2025-12-18T07:17Z — Nested null FPR fix (ticket-01-nested-null-fpr)
- **Data**: synthetic only plus `data/returns_daily.csv` (sha256=96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197) for the real-data smoke.
- **Commands**: `EXEC_MODE=deterministic python experiments/synthetic/nested_killtest.py --config docs/agent_runs/20251218_070342_ticket-01_nested-null-fpr/config.pre.null.yaml --out reports/synthetic_nested_killtest/pre_null_20251218`; post-fix rerun with `config.post.null.yaml`; post-smoke sanity `--config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic_nested_killtest/post_smoke_20251218`; real-data nested weekly smoke `python experiments/equity_panel/run.py --config docs/agent_runs/20251218_070342_ticket-01_nested-null-fpr/config.real.smoke.yaml --design nested --exec-mode deterministic`; tests via `pytest tests/test_gating.py tests/test_nested_killtest_regression.py` and `make test-fast`.
- **Artifacts**:
  - Pre-fix null: `reports/synthetic_nested_killtest/pre_null_20251218/run.json` (null-only, 220 trials).
  - Post-fix null: `reports/synthetic_nested_killtest/post_null_20251218/run.json` (null-only, 220 trials).
  - Post-fix smoke: `reports/synthetic_nested_killtest/post_smoke_20251218/run.json` (null/moderate/strong, 12 each).
  - Real-data nested weekly smoke: `experiments/equity_panel/outputs_nested_smoke_postfix/` (`run_meta.json`, detection_summary.csv, manifest).
- **Results**:
  - Null FPR collapsed: detection_rate 0.8545 → 0.0 on 220 null trials (`summary.csv` in pre_null_20251218 vs post_null_20251218; skip_reason `no_isolated_spike` recorded, calibration present).
  - Power retained: strong spike detection_rate=1.0, moderate≈0.42 on post_smoke_20251218 (null still 0).
  - Real-data smoke conservative: 0 detections across 10 windows (skip_reason `no_isolated_spike`) using tightened gates (use_tvector on, eps=1.0, off_leak=0.3).

## 2025-11-22T00:56Z — Hetzner RC-lite + calibration refresh (git sha 3db9335)
- **Data**: WRDS daily returns `data/returns_daily.csv` (sha256=96ac7dd3…3197) + FF5+MOM factors `data/factors/ff5mom_daily.csv` (sha256=469d44ad…908ca), verified against registries.
- **Commands**:
  - Env/registry: `make setup`; `python - <<'PY' from pathlib import Path; from data.registry import assert_registered_dataset; assert_registered_dataset(Path('data/returns_daily.csv')); PY`.
  - Fast tests: `make test-fast`.
  - RC-lite eval (deterministic, capped windows): `PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py ... --group-design dow --workers 24 --max-windows 200 --out reports/rc-20251121/dow-tyler` and same for `--group-design vol --shrinker oas --gate-delta-frac-min 0.015 --q2-alignment-min-cos 0.9` (see run_manifest for exact commands); `tools/make_summary.py --rc-dir reports/rc-20251121`.
  - Artifacts: `make rc-lite EXEC_MODE=deterministic` (smoke + crisis panels), `python tools/build_memo.py --config experiments/equity_panel/config.rc.yaml`, `python tools/build_brief.py --config experiments/equity_panel/config.rc.yaml`.
  - Calibration: `HARNESS_TRIALS=800 EXEC_MODE=deterministic make sweep:acceptance`.
  - Final tests: `make test-fast`.
- **Results**:
  - RC-lite (DoW/vol, top-60, 126×21, FF5+MOM, first 200 windows): `dow-tyler` detection≈4.32%, acceptance≈4.32%, ΔMSE(EW)=+1.75e-13, ΔMSE(MV)=−2.54e-14, percent_changed≈100%; `vol-tyler` detection≈4.33%, acceptance≈4.33%, ΔMSE(EW)=−1.05e-13, ΔMSE(MV)=−8.64e-14, percent_changed≈100%. Artifacts in `reports/rc-20251121/` (`run_manifest.json`, `metrics_summary.json`, DM tables, risk/diagnostics, prewhiten telemetry).
  - Synthetic ROC sweep (SCM energy-floor): target FPR 2% with threshold ≈0.108 (Tyler FPR≈8.5% at same cut), parameters {delta=0.5, delta_frac=0.02, eps=0.02, stability_eta=0.4}; average power=1.0 at μ ∈ {4,6,8}. Figures refreshed under `reports/figures/`, `calibration_defaults.json` timestamped `2025-11-21T22:07:47Z`.
  - Memo/brief refreshed (`reports/memo.md`, `reports/brief.md`, `reports/memo_20251122_005614.md`, `reports/brief_20251122_005625.md`); gallery already up-to-date.

## 2025-11-13T04:05Z — vol acceptance tuning + paired AWS runs (feat/vol-acceptance-prewhiten@HEAD)
- **Data**: `data/returns_daily.csv` (`sha256=96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) + `data/factors/ff5mom_daily.csv` (`sha256=469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`); both re‑verified via `python tools/verify_dataset.py …`.
- **Local commands**:
  1. `python tools/verify_dataset.py data/returns_daily.csv --registry data/registry.json`
  2. `python tools/verify_dataset.py data/factors/ff5mom_daily.csv --registry data/factors/registry.json`
  3. `make test-fast` (unit suite: 65 pass / 140 deselected).
  4. `INSTANCE_DNS=… KEY_PATH=… make aws:rc-vol AWS_ARGS='USE_FACTORS=0 EXEC_MODE=deterministic RC_PROGRESS=1 RC_GATE_DELTA_FRAC_MIN=0.015 RC_VOL_MIN_REPS=10 RC_VOL_GROUP_REPS=6 Q_MAX_VOL=2 RC_VOL_ASSETS=80'`
  5. `INSTANCE_DNS=… KEY_PATH=… make aws:rc-vol AWS_ARGS='USE_FACTORS=1 EXEC_MODE=deterministic RC_PROGRESS=1 RC_GATE_DELTA_FRAC_MIN=0.015 RC_VOL_MIN_REPS=10 RC_VOL_GROUP_REPS=6 Q_MAX_VOL=2 RC_VOL_ASSETS=80'`
  6. `python tools/prewhiten_effect.py --off reports/rc-20251113/vol-off --on reports/rc-20251113/vol-ff5mom --mirror`
  7. `make gallery` and `make memo`
- **Artifacts**:
  - Run metadata: `reports/runs/20251112T232711Z/` (vol-off) and `reports/runs/20251113T014417Z/` (vol-ff5mom) with `make_*.log`, telemetry (`metrics*.json[ln]`), `run.json`, etc.
  - Bounded RC outputs (top‑80, 126×21) under `reports/rc-20251113/{vol-off,vol-ff5mom}/` including `metrics.csv`, `risk.csv`, `diagnostics.csv`, `dm.csv`, `dm_flip_only.csv`, `flip_dm.png`, and mirrored `prewhiten_effect.csv`.
  - Manifest + doc updates: `reports/rc-20251113/run_manifest.json`, refreshed `reports/memo.md` + timestamped `reports/memo_20251113_040459.md`, README/REPORT sections describing the new knobs + reporting stack.
- **Results**:
  - Acceptance (full regime) landed inside the 2–6 % band: off run `acceptance_rate=2.38 %`, FF5+MOM `acceptance_rate=2.26 %` with substitution fractions ≈1.6 % and ≈1.5 % respectively on the top‑80 universe (`reports/rc-20251113/*/full/diagnostics.csv`).
  - Flip-set telemetry: `reports/rc-20251113/vol-off/dm_flip_only.csv` shows `n_effective=110` (stats are NaN because residuals match baseline), while the prewhitened run logs `n_effective=117` with sign-test stats (`ew vs baseline`: z≈5.57, p≈9.3e‑10; `mv vs baseline`: z≈3.36, p≈9.8e‑4).
  - `reports/rc-20251113/vol-ff5mom/prewhiten_effect.csv` summarises the paired deltas: detection_rate +3.6 bps, ΔMSE(EW)=+4.1e‑11, ΔMSE(MV)=‑3.0e‑12, ES95 errors tighten by ≈0.88 bps, and the sign-test p‑values above confirm the flip-set improvement.

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

## 2025-11-21T08:15Z — Nested guardrails + RC-lite refresh (a94ad00)
- **Data**: WRDS `data/returns_daily.csv` + `data/factors/ff5mom_daily.csv` (registry hashes unchanged).
- **Commands**:
  - Nested smoke (throughput, tuned guard): `.venv/bin/python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --no-progress --workers 32 --assets-top 80 --stride-windows 4 --cache-dir .cache --precompute-panel --drop-partial-weeks --estimator dealias --prewhiten ff5mom --use-factor-prewhiten 1` (detection_rate=1/24).
  - RC-lite (WRDS): `PATH=.venv/bin:$PATH RC_WORKERS=$(nproc) EXEC_MODE=throughput make rc-lite` followed by `make memo` to rebuild gallery/memos (latest: `reports/memo_20251121_081543.md`).
  - Acceptance sweep: `PATH=.venv/bin:$PATH EXEC_MODE=throughput HARNESS_TRIALS=400 make sweep:acceptance` (calibration_defaults.json regenerated; ROC figs under `reports/figures/`).
  - Ablations: `python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml` (assets_top=60, 2020–2021 slice) and `.venv/bin/python experiments/equity_panel/run.py --config experiments/equity_panel/config.ablation.smoke.yaml --no-progress --workers 8 --assets-top 60 --stride-windows 4 --cache-dir .cache --precompute-panel --drop-partial-weeks --estimator dealias --prewhiten ff5mom --use-factor-prewhiten 1 --ablations`.
  - Tests: `PATH=.venv/bin:$PATH make test-fast` (65 passed).
- **Results**:
  - Nested coverage now in-band: `experiments/equity_panel/outputs_nested_smoke/.../summary.json` shows detection_windows=1/24 (4.17%) with `nested_guard` skips=5 after applying stability/edge floor of 3 bps.
  - RC-lite artifacts refreshed in place (`figures/rc`, memo at `reports/memo_20251121_081543.md`); nested plots reflect the new guardrails.
  - Synthetic acceptance defaults re-written (timestamp 2025-11-21T03:30Z) via sweep; thresholds unchanged numerically (delta_frac=0.02, eta=0.4, energy_floor≈0.1012).
  - Ablation assets/regime aligned to RC (2020–2021 calm+crisis): updated grid in `experiments/ablate/ablation_matrix_tiny.yaml`, matrix at `ablations/ablation_matrix.csv`, and E5 summary at `experiments/equity_panel/outputs_ablation_smoke/oneway_J5.../ablation_summary.csv`.

## 2025-11-20T23:43Z — Nested gating tweak + ablation plumbing (68c1c6d)
- **Data**: WRDS daily returns `data/returns_daily.csv` and factors `data/factors/ff5mom_daily.csv` (hashes unchanged).
- **Commands**: 
  - `.venv` bootstrap (`python3 -m venv .venv && pip install -e .[dev]`), `make test-fast` (65 passed).
  - Nested smoke reruns: `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --no-progress --exec-mode deterministic --factor-csv data/factors/ff5mom_daily.csv --prewhiten ff5mom --use-factor-prewhiten 1 --estimator dealias` (multiple passes after gating relax/threshold tweaks).
  - Ablation attempts: `make rc-ablations` and `python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml` (timed out on this host after ≥10 min per attempt; no new ablation artifacts written yet).
- **Results/Notes**: Nested gating now records non-isolated fallback telemetry and is enabled via config, but the current nested smoke still reports 0/24 detection windows (no candidates emitted by `dealias_search`). Ablation grid defaults switched to the tiny matrix and hooked into `rc-ablations`, but runs remain long locally; expect faster completion on Hetzner once queued. No changes to calibration files. Next steps: run ablate tiny matrix on Hetzner or further shrink window/asset subset; investigate why nested detection remains zero despite relaxed stability/delta and non-isolated fallback.

## 2025-11-21T02:10Z — Hetzner ablations + nested coverage unlocked (d1b39ed)
- **Data**: Same WRDS returns/factors as prior entries.
- **Commands**:
  - Nested smoke (throughput): `PYTHONPATH=src OMP_NUM_THREADS=4 EXEC_MODE=throughput python experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --no-progress --exec-mode throughput --factor-csv data/factors/ff5mom_daily.csv --prewhiten ff5mom --use-factor-prewhiten 1 --estimator dealias`.
  - Ablation matrix (tiny grid): `EXEC_MODE=throughput python experiments/ablate/run.py --config experiments/ablate/ablation_matrix_tiny.yaml`.
  - Fast E5 ablation drop (direct call to _run_param_ablation): inline Python snippet loading `data/returns_daily.csv` (2023-01-03→2023-02-10) and writing `experiments/equity_panel/outputs_ablation_smoke/ablation_summary.csv`.
  - Gallery/memo refresh: `python tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml`; `python tools/build_memo.py --config experiments/equity_panel/config.rc.yaml`.
  - Tests: `.venv` active; `make test-fast` (65 passed).
- **Results**:
  - Nested smoke now logs 7/24 detection windows (29% coverage) with gating skips recorded; edge margin median ~0.023; non-isolated fallback not triggered; summary at `experiments/equity_panel/outputs_nested_smoke/.../summary.json`.
  - Ablation artifacts: `ablations/ablation_matrix.csv` (4-combo tiny grid) and `experiments/equity_panel/outputs_ablation_smoke/ablation_summary.csv`; gallery/memo pick up the matrix (heatmap/table) and ablation summary directory.
- **Notes**: Added `use_tvector` toggle (configurable, disabled for nested smoke and ablations) to bypass overly strict t-vector gating; relaxed nested thresholds (delta_frac 0.005, eps 0.01, eta 0.15, q_max 2, require_isolated=false). Equity-panel ablation runner still heavy when invoked via `rc-ablations`; direct `_run_param_ablation` was used to emit the E5 summary for this drop. Next: rerun the full `rc-ablations` target on Hetzner if time permits, or wire timeout/limit guards.
