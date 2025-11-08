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
RC_PROGRESS ?= 0
RC_WORKERS := $(shell python3 -c 'import os;print(os.cpu_count() or 4)')
RC_FLAGS_BASE := --workers $(RC_WORKERS) --assets-top 100 --stride-windows 4 --resume --cache-dir .cache --precompute-panel --drop-partial-weeks --oneway-a-solver auto
RC_FLAGS := $(RC_FLAGS_BASE)
ifeq ($(RC_PROGRESS),0)
RC_FLAGS := --no-progress $(RC_FLAGS)
endif
RC_RETURNS := data/returns_daily.csv
RC_DATE := $(shell python3 -c 'import datetime as _dt; print(_dt.datetime.utcnow().strftime("%Y%m%d"))')
RC_OUT := reports/rc-$(RC_DATE)
RC_REGISTRY := data/registry.json
RC_VERIFY_DATASET := python tools/verify_dataset.py $(RC_RETURNS) --registry $(RC_REGISTRY)
RC_FACTORS ?= data/factors/ff5mom_daily.csv
RC_FACTORS_REGISTRY ?= data/factors/registry.json
RC_VERIFY_FACTORS := python tools/verify_dataset.py $(RC_FACTORS) --registry $(RC_FACTORS_REGISTRY)
RC_GATE_CALIB := calibration/edge_delta_thresholds.json
RC_GATE_DEFAULTS := calibration/defaults.json
RC_WINDOW ?= 126
RC_HORIZON ?= 21
RC_START ?= 2018-01-01
RC_END ?= 2024-12-31
RC_GATE_DELTA_FRAC ?= 0.02
RC_MV_GAMMA ?= 1e-4
RC_MV_BOX ?= 0.0,0.1
RC_MV_TURNOVER_BPS ?= 5
RC_MV_CONDITION_CAP ?= 1000000
ABLA_GRID ?= experiments/ablate/ablation_matrix.yaml
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
	$(RC_PY) experiments/eval/run.py --returns-csv $(RC_RETURNS) --out $(RC_OUT)

rc-ablations:
	$(RC_PY) experiments/ablate/run.py --config $(ABLA_GRID) $(if $(RC_CALM_WINDOW_SAMPLE),--calm-window-sample $(RC_CALM_WINDOW_SAMPLE),) $(if $(RC_CRISIS_WINDOW_TOPK),--crisis-window-topk $(RC_CRISIS_WINDOW_TOPK),)

rc-summary:
	$(RC_PY) tools/make_summary.py --rc-dir $(RC_OUT)

rc: rc-data rc-eval
	$(MAKE) rc-ablations
	$(MAKE) rc-summary RC_OUT=$(RC_OUT)
	$(MAKE) memo

rc-lite:
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator dealias
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator lw
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml $(RC_FLAGS) --estimator oas
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS) --estimator dealias
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS) --estimator lw
	$(RC_PY) experiments/equity_panel/run.py --config experiments/equity_panel/config.crisis.2020.yaml $(RC_FLAGS) --estimator oas
	$(RC_PY) tools/build_gallery.py --config experiments/equity_panel/config.rc.yaml
	$(RC_PY) tools/build_memo.py --config experiments/equity_panel/config.rc.yaml

.PHONY: aws\:rc-lite aws\:rc aws\:sweep-calibration aws\:rc-sensitivity
AWS_ARGS ?=

aws\:%:
	scripts/aws_run.sh $* $(AWS_ARGS)

DOW_EDGE := $(if $(EDGE),$(EDGE),tyler)
VOL_EDGE := $(if $(EDGE),$(EDGE),tyler)
RC_DOW_OUT := $(RC_OUT)/dow-$(DOW_EDGE)
RC_VOL_OUT := $(RC_OUT)/vol-$(VOL_EDGE)
RC_DOW_ASSETS ?= 60
RC_VOL_ASSETS ?= 80
RC_DOW_SHRINKER ?= rie
RC_VOL_SHRINKER ?= oas
RC_DOW_PREWHITEN ?= ff5mom
RC_VOL_PREWHITEN ?= ff5mom
RC_VOL_GROUP_MIN ?= 3
RC_VOL_GROUP_REPS ?= 10
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
		--edge-mode $(DOW_EDGE) \
		--shrinker $(RC_DOW_SHRINKER) \
		--prewhiten $(RC_DOW_PREWHITEN) \
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC) \
		--require-isolated \
		--q-max 1 \
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
		--edge-mode $(VOL_EDGE) \
		--shrinker $(RC_VOL_SHRINKER) \
		--prewhiten $(RC_VOL_PREWHITEN) \
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC) \
		--require-isolated \
		--q-max 1 \
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
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC) \
		--require-isolated \
		--q-max 1 \
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
		$(if $(USE_FACTORS),--use-factor-prewhiten $(USE_FACTORS),) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC) \
		--require-isolated \
		--q-max 1 \
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

.PHONY: inject-spike
inject-spike:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	PYTHONPATH=src python experiments/eval/inject_spike.py \
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
	PYTHONPATH=src python experiments/synthetic/calibrate_thresholds.py \
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
