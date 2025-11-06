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

.PHONY: aws\:rc-lite aws\:rc aws\:sweep-calibration
AWS_ARGS ?=

aws\:%:
	scripts/aws_run.sh $* $(AWS_ARGS)

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
