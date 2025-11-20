.PHONY: setup fmt lint test run-synth run-equity

HARNESS_TRIALS ?= 400

setup:
	pip install --upgrade pip
	pip install -e '.[dev]'

.PHONY: env
env: setup

fmt:
	black src experiments tests
	ruff check --fix src experiments tests

lint:
	ruff check src experiments tests
	mypy src

test:
	pytest -q

.PHONY: test-fast test-integration test-slow test-all
test-fast:
	pytest -m "unit"

test-integration:
	pytest -m "integration"

test-slow:
	pytest -m "slow"

test-all:
	pytest -m "unit or integration"

.PHONY: smoke-daily
smoke-daily:
	PYTHONPATH=src python experiments/daily/run.py --returns-csv data/returns_daily.csv --design dow --window 60 --horizon 10 --out reports/smoke-daily/dow
	PYTHONPATH=src python experiments/daily/run.py --returns-csv data/returns_daily.csv --design vol --window 60 --horizon 10 --out reports/smoke-daily/vol --shrinker quest

.PHONY: test-progress
test-progress:
	# Verbose output with progress bar (pytest-sugar), parallel if available
	pytest -v -n auto || pytest -v

.PHONY: gallery memo report rc rc-data rc-lite rc-eval rc-summary rc-ablations
gallery:
	$(RC_PY) tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml

memo: gallery
	$(RC_PY) tools/build_memo.py --config experiments/equity_panel/config.rc.yaml

report: gallery

RC_PY := PYTHONPATH=src:. OMP_NUM_THREADS=1 python3
USE_FACTORS ?= 1
RC_PROGRESS ?= 0
RC_WORKERS := $(shell python3 -c 'import os;print(os.cpu_count() or 4)')
RC_RETURNS := data/returns_daily.csv
RC_DATE := $(shell python3 -c 'import datetime as _dt; print(_dt.datetime.utcnow().strftime("%Y%m%d"))')
RC_OUT := reports/rc-$(RC_DATE)
RC_REGISTRY := data/registry.json
RC_VERIFY_DATASET := python tools/verify_dataset.py $(RC_RETURNS) --registry $(RC_REGISTRY)
RC_FACTORS ?= data/factors/ff5mom_daily.csv
RC_FACTORS_REGISTRY ?= data/factors/registry.json
RC_VERIFY_FACTORS := python tools/verify_dataset.py $(RC_FACTORS) --registry $(RC_FACTORS_REGISTRY)
RC_PREWHITEN ?= ff5mom
RC_USE_FACTOR_PREWHITEN ?= 1
RC_OVERLAY_DELTA ?= 0.05
RC_COARSE_CANDIDATE ?= 1
RC_GATE_ACCEPT_NONISOLATED ?= 1
RC_GATE_STABILITY_MIN ?= 0.0001
RC_REQUIRE_ISOLATED ?= 1
RC_DOW_MIN_REPS ?= 10
RC_VOL_MIN_REPS ?= 10
RC_VOL_REQUIRE_ISOLATED ?= 0
ifeq ($(RC_REQUIRE_ISOLATED),1)
RC_ISOLATION_FLAG := --require-isolated
else
RC_ISOLATION_FLAG := --allow-non-isolated
endif
ifeq ($(RC_VOL_REQUIRE_ISOLATED),1)
RC_VOL_ISOLATION_FLAG := --require-isolated
else
RC_VOL_ISOLATION_FLAG := --allow-non-isolated
endif
RC_FLAGS_BASE := --workers $(RC_WORKERS) --assets-top 100 --stride-windows 4 --resume --cache-dir .cache --precompute-panel --drop-partial-weeks --oneway-a-solver auto --factor-csv $(RC_FACTORS) --prewhiten $(RC_PREWHITEN) --use-factor-prewhiten $(RC_USE_FACTOR_PREWHITEN)
RC_FLAGS := $(RC_FLAGS_BASE)
ifeq ($(RC_PROGRESS),0)
RC_FLAGS := --no-progress $(RC_FLAGS)
endif
RC_GATE_CALIB := calibration/edge_delta_thresholds.json
RC_GATE_DEFAULTS := calibration/defaults.json
RC_GATE_MODE ?= soft
RC_WINDOW ?= 126
RC_HORIZON ?= 21
RC_START ?= 2018-01-01
RC_END ?= 2024-12-31
RC_GATE_DELTA_FRAC_MIN ?= 0.02
RC_GATE_DELTA_FRAC_MIN_VOL ?= 0.015
RC_Q_MAX ?= 2
Q_MAX_VOL ?= 2
VOL_Q2_ALIGNMENT_MIN_COS ?= 0.9
RC_MV_GAMMA ?= 1e-4
RC_MV_BOX ?= 0.0,0.1
RC_MV_TURNOVER_BPS ?= 5
RC_MV_CONDITION_CAP ?= 1000000
ABLA_GRID ?= experiments/ablate/ablation_matrix_tiny.yaml
RC_CALM_WINDOW_SAMPLE ?=
RC_CRISIS_WINDOW_TOPK ?=

CALIB_P_ASSETS ?= 64 80 96
CALIB_N_GROUPS ?= 36
CALIB_REPLICATES ?= 14 20
CALIB_REPLICATE_BINS ?= r12-16:12-16 r17-22:17-22
CALIB_ASSET_BINS ?= p64-96:64-96
CALIB_DELTA_ABS ?= 0.35 0.45 0.55 0.65
CALIB_DELTA_FRAC ?= 0.01 0.015 0.02 0.025 0.03
CALIB_STABILITY ?= 0.30 0.40 0.50 0.60
CALIB_ALPHA ?= 0.02
CALIB_TRIALS_NULL ?= 300
CALIB_TRIALS_ALT ?= 200
CALIB_WORKERS ?= $(shell python3 -c 'import os;print(os.cpu_count() or 8)')
CALIB_BATCH_SIZE ?= 100
MP_CACHE_DIR ?= .cache/mp_edges
EXEC_MODE ?= deterministic

rc-data: 
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator dealias
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator lw
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator oas
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator cc
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator factor
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator tyler_shrink
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml $(RC_FLAGS) --estimator dealias
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS)
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2022.yaml $(RC_FLAGS)

rc-eval:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/run.py --returns-csv $(RC_RETURNS) --factors-csv $(RC_FACTORS) --prewhiten $(RC_PREWHITEN) --use-factor-prewhiten $(RC_USE_FACTOR_PREWHITEN) --overlay-delta $(RC_OVERLAY_DELTA) --coarse-candidate $(RC_COARSE_CANDIDATE) --gate-mode $(RC_GATE_MODE) $(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) $(if $(RC_GATE_STABILITY_MIN),--gate-stability-min $(RC_GATE_STABILITY_MIN),) --out $(RC_OUT)

rc-ablations:
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.ablation.smoke.yaml $(RC_FLAGS) --estimator dealias --ablations
	$(RC_PY) experiments/ablate/run.py --config $(ABLA_GRID) $(if $(RC_CALM_WINDOW_SAMPLE),--calm-window-sample $(RC_CALM_WINDOW_SAMPLE),) $(if $(RC_CRISIS_WINDOW_TOPK),--crisis-window-topk $(RC_CRISIS_WINDOW_TOPK),)

rc-summary:
	$(RC_PY) tools/make_summary.py --rc-dir $(RC_OUT)

rc: rc-data rc-eval
	$(MAKE) rc-ablations
	$(MAKE) rc-summary RC_OUT=$(RC_OUT)
	$(MAKE) memo

rc-lite:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator dealias
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator lw
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator oas
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS) --estimator dealias
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS) --estimator lw
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS) --estimator oas
	$(RC_PY) tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml
	$(RC_PY) tools/build_memo.py --config experiments/equity_panel/config.rc.yaml

.PHONY: rc-lite-sanity
rc-lite-sanity:
	$(MAKE) rc-dow
	$(MAKE) rc-vol
	$(RC_PY) tools/make_summary.py --rc-dir $(RC_OUT)
	python3 - <<'PY'
	import json
	from pathlib import Path

	import pandas as pd

	root = Path("$(RC_OUT)").resolve()
	entries = {
	    "dow": Path("$(RC_DOW_OUT)").resolve(),
	    "vol": Path("$(RC_VOL_OUT)").resolve(),
	}
	summary = {"rc_dir": str(root), "entries": {}}
	regime_frames = []
	for label, subdir in entries.items():
	    diag_path = subdir / "diagnostics.csv"
	    record = {"path": str(subdir)}
	    if diag_path.exists():
	        diag_df = pd.read_csv(diag_path)
	        if not diag_df.empty:
	            row = diag_df.iloc[0]
	            record["detection_rate"] = float(row.get("detection_rate", float("nan")))
	            record["alignment_cos"] = float(row.get("alignment_cos_mean", float("nan")))
	            record["reason_code"] = row.get("reason_code", "")
	    summary["entries"][label] = record
	    regime_path = subdir / "regime.csv"
	    if regime_path.exists():
	        regime_df = pd.read_csv(regime_path)
	        regime_df.insert(0, "design", label)
	        regime_frames.append(regime_df)

	regime_out = root / "regime.csv"
	if regime_frames:
	    pd.concat(regime_frames, ignore_index=True).to_csv(regime_out, index=False)
	else:
	    pd.DataFrame(columns=["design"]).to_csv(regime_out, index=False)
	summary_path = root / "summary_sanity.json"
	summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
	print(f"[rc-lite-sanity] Wrote {summary_path}")
		print(f"[rc-lite-sanity] Wrote {regime_out}")
	PY

.PHONY: aws\:rc-lite aws\:rc aws\:sweep-calibration aws\:rc-sensitivity
AWS_ARGS ?=

aws\:%:
	scripts/aws_run.sh $* $(AWS_ARGS)

DOW_EDGE := $(if $(EDGE),$(EDGE),tyler)
VOL_EDGE := $(if $(EDGE),$(EDGE),tyler)
RC_DOW_OUT := $(RC_OUT)/dow-$(DOW_EDGE)
RC_VOL_OUT := $(RC_OUT)/vol-$(VOL_EDGE)
RC_DOW_ASSETS ?= 60
RC_VOL_ASSETS ?= 60
RC_DOW_SHRINKER ?= rie
RC_VOL_SHRINKER ?= oas
RC_DOW_PREWHITEN ?= ff5mom
RC_VOL_PREWHITEN ?= ff5mom
RC_DOW_GROUP_MIN ?= 5
RC_DOW_GROUP_REPS ?= 3
RC_DOW_MIN_REPS ?= 10
RC_VOL_MIN_REPS ?= 10
RC_VOL_GROUP_MIN ?= 3
RC_VOL_GROUP_REPS ?= 6
RC_WEEK_OUT := $(RC_OUT)/week
RC_WEEK_ASSETS ?= 80
RC_WEEK_GROUP_MIN ?= 4
RC_WEEK_GROUP_REPS ?= 5
RC_WEEK_SHRINKER ?= $(RC_DOW_SHRINKER)
RC_WEEK_PREWHITEN ?= $(RC_DOW_PREWHITEN)
RC_DOWXVOL_OUT := $(RC_OUT)/dowxvol
RC_DOWXVOL_ASSETS ?= 90
RC_DOWXVOL_GROUP_MIN ?= 10
RC_DOWXVOL_GROUP_REPS ?= 3
RC_DOWXVOL_SHRINKER ?= $(RC_DOW_SHRINKER)
RC_DOWXVOL_PREWHITEN ?= $(RC_DOW_PREWHITEN)
RC_SENS_START ?= 2024-05-01
RC_SENS_END ?= 2024-10-31
RC_SENS_LABEL ?= rc-sensitivity-$(RC_DATE)
RC_INJECT_OUT ?= reports/figures

.PHONY: rc-dow rc-vol rc-week rc-dowxvol
rc-dow:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/run.py \
		--returns-csv $(RC_RETURNS) \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--start $(RC_START) \
		--end $(RC_END) \
		--assets-top $(RC_DOW_ASSETS) \
		--group-design dow \
		--group-min-count $(RC_DOW_GROUP_MIN) \
		--group-min-replicates $(RC_DOW_GROUP_REPS) \
		--min-reps-dow $(RC_DOW_MIN_REPS) \
		--edge-mode $(DOW_EDGE) \
		--shrinker $(RC_DOW_SHRINKER) \
		--prewhiten $(RC_DOW_PREWHITEN) \
		--overlay-delta $(RC_OVERLAY_DELTA) \
		--coarse-candidate $(RC_COARSE_CANDIDATE) \
		--gate-mode $(RC_GATE_MODE) \
		$(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) \
		$(if $(RC_GATE_STABILITY_MIN),--gate-stability-min $(RC_GATE_STABILITY_MIN),) \
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC_MIN) \
		$(RC_ISOLATION_FLAG) \
		--min-reps-dow $(RC_DOW_MIN_REPS) \
		--q-max $(RC_Q_MAX) \
		--mv-gamma $(RC_MV_GAMMA) \
		--mv-box $(RC_MV_BOX) \
		--mv-turnover-bps $(RC_MV_TURNOVER_BPS) \
		--mv-condition-cap $(RC_MV_CONDITION_CAP) \
		--out $(RC_DOW_OUT)

rc-vol:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/run.py \
		--returns-csv $(RC_RETURNS) \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--start $(RC_START) \
		--end $(RC_END) \
		--assets-top $(RC_VOL_ASSETS) \
		--group-design vol \
		--group-min-count $(RC_VOL_GROUP_MIN) \
		--group-min-replicates $(RC_VOL_GROUP_REPS) \
		--min-reps-vol $(RC_VOL_MIN_REPS) \
		--edge-mode $(VOL_EDGE) \
		--shrinker $(RC_VOL_SHRINKER) \
		--prewhiten $(RC_VOL_PREWHITEN) \
		--overlay-delta $(RC_OVERLAY_DELTA) \
		--coarse-candidate $(RC_COARSE_CANDIDATE) \
		--gate-mode $(RC_GATE_MODE) \
		$(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) \
		$(if $(RC_GATE_STABILITY_MIN),--gate-stability-min $(RC_GATE_STABILITY_MIN),) \
		$(RC_VOL_ISOLATION_FLAG) \
		--min-reps-vol $(RC_VOL_MIN_REPS) \
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC_MIN_VOL) \
		--q-max $(Q_MAX_VOL) \
		$(if $(VOL_Q2_ALIGNMENT_MIN_COS),--q2-alignment-min-cos $(VOL_Q2_ALIGNMENT_MIN_COS),) \
		--mv-gamma $(RC_MV_GAMMA) \
		--mv-box $(RC_MV_BOX) \
		--mv-turnover-bps $(RC_MV_TURNOVER_BPS) \
		--mv-condition-cap $(RC_MV_CONDITION_CAP) \
		--out $(RC_VOL_OUT)

rc-week:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/run.py \
		--returns-csv $(RC_RETURNS) \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--start $(RC_START) \
		--end $(RC_END) \
		--assets-top $(RC_WEEK_ASSETS) \
		--group-design week \
		--group-min-count $(RC_WEEK_GROUP_MIN) \
		--group-min-replicates $(RC_WEEK_GROUP_REPS) \
		--edge-mode $(DOW_EDGE) \
		--shrinker $(RC_WEEK_SHRINKER) \
		--prewhiten $(RC_WEEK_PREWHITEN) \
		--overlay-delta $(RC_OVERLAY_DELTA) \
		--coarse-candidate $(RC_COARSE_CANDIDATE) \
		--gate-mode $(RC_GATE_MODE) \
		$(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) \
		$(if $(RC_GATE_STABILITY_MIN),--gate-stability-min $(RC_GATE_STABILITY_MIN),) \
		$(RC_ISOLATION_FLAG) \
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC_MIN) \
		--q-max $(RC_Q_MAX) \
		--mv-gamma $(RC_MV_GAMMA) \
		--mv-box $(RC_MV_BOX) \
		--mv-turnover-bps $(RC_MV_TURNOVER_BPS) \
		--mv-condition-cap $(RC_MV_CONDITION_CAP) \
		--out $(RC_WEEK_OUT)

rc-dowxvol:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/run.py \
		--returns-csv $(RC_RETURNS) \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--start $(RC_START) \
		--end $(RC_END) \
		--assets-top $(RC_DOWXVOL_ASSETS) \
		--group-design dowxvol \
		--group-min-count $(RC_DOWXVOL_GROUP_MIN) \
		--group-min-replicates $(RC_DOWXVOL_GROUP_REPS) \
		--edge-mode $(DOW_EDGE) \
		--shrinker $(RC_DOWXVOL_SHRINKER) \
		--prewhiten $(RC_DOWXVOL_PREWHITEN) \
		--overlay-delta $(RC_OVERLAY_DELTA) \
		--coarse-candidate $(RC_COARSE_CANDIDATE) \
		--gate-mode $(RC_GATE_MODE) \
		$(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) \
		$(if $(RC_GATE_STABILITY_MIN),--gate-stability-min $(RC_GATE_STABILITY_MIN),) \
		$(RC_ISOLATION_FLAG) \
		--min-reps-vol $(RC_VOL_MIN_REPS) \
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC_MIN) \
		--q-max $(RC_Q_MAX) \
		--mv-gamma $(RC_MV_GAMMA) \
		--mv-box $(RC_MV_BOX) \
		--mv-turnover-bps $(RC_MV_TURNOVER_BPS) \
		--mv-condition-cap $(RC_MV_CONDITION_CAP) \
		--out $(RC_DOWXVOL_OUT)

.PHONY: rc-sensitivity
rc-sensitivity:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/sensitivity.py \
		--returns-csv $(RC_RETURNS) \
		--slice-start $(RC_SENS_START) \
		--slice-end $(RC_SENS_END) \
		--assets-top 150 \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--config experiments/eval/config.yaml \
		--thresholds experiments/eval/thresholds.json \
		--registry $(RC_REGISTRY) \
		--out reports/rc-sensitivity \
		--label $(RC_SENS_LABEL) \
		--workers 1

.PHONY: rc-sensitivity-coarse
rc-sensitivity-coarse:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/sensitivity.py \
		--returns-csv $(RC_RETURNS) \
		--slice-start $(RC_SENS_START) \
		--slice-end $(RC_SENS_END) \
		--assets-top 150 \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--config experiments/eval/config.yaml \
		--thresholds experiments/eval/thresholds.json \
		--registry $(RC_REGISTRY) \
		--out reports/rc-sensitivity \
		--label $(RC_SENS_LABEL)-coarse \
		--workers 1 \
		--coarse-candidate 1

.PHONY: inject-spike
inject-spike:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/inject_spike.py \
		--returns-csv $(RC_RETURNS) \
		--factors-csv $(RC_FACTORS) \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--start $(RC_START) \
		--end $(RC_END) \
		--assets-top 150 \
		--config experiments/eval/config.yaml \
		--thresholds experiments/eval/thresholds.json \
		--group-design week \
		--use-factor-prewhiten 1 \
		--out $(RC_INJECT_OUT)

.PHONY: inject-spike-coarse
inject-spike-coarse:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	$(RC_PY) experiments/eval/inject_spike.py \
		--returns-csv $(RC_RETURNS) \
		--factors-csv $(RC_FACTORS) \
		--window $(RC_WINDOW) \
		--horizon $(RC_HORIZON) \
		--start $(RC_START) \
		--end $(RC_END) \
		--assets-top 150 \
		--config experiments/eval/config.yaml \
		--thresholds experiments/eval/thresholds.json \
		--group-design week \
		--use-factor-prewhiten 1 \
		--coarse-candidate 1 \
		--out $(RC_INJECT_OUT)

run-synth:
	python experiments/synthetic_oneway/run.py

run-equity:
	python experiments/equity_panel/run.py

.PHONY: run\:equity_smoke
run\:equity_smoke:
	PYTHONPATH=src python experiments/equity_panel/run.py \
		--config experiments/equity_panel/config.smoke.yaml \
		--gating-mode fixed \
		--minvar-ridge 0.0001 \
		--minvar-box 0.0,0.1 \
		--minvar-condition-cap 1000000000 \
		--turnover-cost 5

.PHONY: sweep\:acceptance
sweep\:acceptance:
	PYTHONPATH=src python experiments/synthetic/null.py \
		--trials $(HARNESS_TRIALS) \
		--out reports/synthetic/null_harness \
		--figures-out reports/figures
	PYTHONPATH=src python experiments/synthetic/power.py \
		--trials $(HARNESS_TRIALS) \
		--null-scores reports/synthetic/null_harness/null_scores.parquet \
		--out reports/synthetic/power_harness \
		--figures-out reports/figures \
		--defaults-path calibration_defaults.json

.PHONY: sweep-calibration
sweep-calibration: sweep\:acceptance

.PHONY: calibrate-thresholds
calibrate-thresholds:
	PYTHONPATH=src:. python experiments/synthetic/calibrate_thresholds.py \
		--alpha $(CALIB_ALPHA) \
		--p-assets $(CALIB_P_ASSETS) \
		--n-groups $(CALIB_N_GROUPS) \
		--replicates $(CALIB_REPLICATES) \
		--replicate-bins $(CALIB_REPLICATE_BINS) \
		--asset-bins $(CALIB_ASSET_BINS) \
		--trials-null $(CALIB_TRIALS_NULL) \
		--trials-alt $(CALIB_TRIALS_ALT) \
		--delta-abs-grid $(CALIB_DELTA_ABS) \
		--delta-frac-grid $(CALIB_DELTA_FRAC) \
		--stability-grid $(CALIB_STABILITY) \
		--edge-modes scm tyler \
		--workers $(CALIB_WORKERS) \
		--batch-size $(CALIB_BATCH_SIZE) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) \
		$(if $(SHARD_MANIFEST),--shard-manifest $(SHARD_MANIFEST),) \
		$(if $(SHARD_ID),--shard-id $(SHARD_ID),) \
		--exec-mode $(EXEC_MODE) \
		--mp-cache-dir $(MP_CACHE_DIR) \
		--verbose \
		--out calibration/edge_delta_thresholds.json \
		--defaults-out calibration/defaults.json

.PHONY: run-equity-crisis
run-equity-crisis:
	python3 experiments/equity_panel/run.py \
	  --crisis 2020-02-01:2020-05-31 \
	  --delta-frac 0.03 --eps 0.03 --a-grid 180 --eta 0.4 --signed-a \
	  --window-weeks 8 --horizon-weeks 2

.PHONY: figures
figures: ## regenerate all figures (synthetic + equity)
	python experiments/synthetic_oneway/run.py
	python experiments/equity_panel/run.py

.PHONY: bench-linalg
bench-linalg:
	python scripts/bench_linalg.py
