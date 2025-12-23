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

.PHONY: gallery memo report rc rc-data rc-lite rc-eval rc-summary rc-ablations rc-paper-v1-ablate
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
RC_LITE_STAMP := $(shell date +%Y%m%d_%H%M%S)
RC_LITE_BASE := experiments/equity_panel/outputs_rc-lite-$(RC_DATE)_$(RC_LITE_STAMP)
RC_LITE_CACHE := .cache/rc-lite
RC_OUT_SANITY := $(RC_OUT)-sanity-$(RC_LITE_STAMP)
RC_WEEKLY_DOW_OUT := $(RC_LITE_BASE)/dow-weekly
RC_WEEKLY_NESTED_OUT := $(RC_LITE_BASE)/nested
RC_SMOKE_CONFIG := experiments/equity_panel/config.smoke.yaml
RC_NESTED_SMOKE_CONFIG := experiments/equity_panel/config.nested.smoke.yaml
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
PAPER_V1_CONFIG := experiments/eval/config.paper_v1.yaml
PAPER_V1_ABLATE_STAMP := $(shell date +%Y%m%d_%H%M%S)
PAPER_V1_ABLATE_ROOT := reports/rc-paper-v1-ablate-$(PAPER_V1_ABLATE_STAMP)
PAPER_V1_RETURNS ?= data/returns_daily.csv
PAPER_V1_FACTORS ?= data/factors/ff5mom_daily.csv
PAPER_V1_OFF_FLAGS := --q-max 0
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
	$(RC_PY) tools/make_summary.py --rc-dir $(RC_OUT_SANITY)

rc-paper-v1-ablate:
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	mkdir -p $(PAPER_V1_ABLATE_ROOT)
	$(RC_PY) -m experiments.eval.run --config $(PAPER_V1_CONFIG) --returns-csv $(PAPER_V1_RETURNS) --factors-csv $(PAPER_V1_FACTORS) --shrinker sample --out $(PAPER_V1_ABLATE_ROOT)/scm_off $(PAPER_V1_OFF_FLAGS)
	$(RC_PY) -m experiments.eval.run --config $(PAPER_V1_CONFIG) --returns-csv $(PAPER_V1_RETURNS) --factors-csv $(PAPER_V1_FACTORS) --shrinker sample --out $(PAPER_V1_ABLATE_ROOT)/scm_on
	$(RC_PY) -m experiments.eval.run --config $(PAPER_V1_CONFIG) --returns-csv $(PAPER_V1_RETURNS) --factors-csv $(PAPER_V1_FACTORS) --shrinker oas --out $(PAPER_V1_ABLATE_ROOT)/oas_off $(PAPER_V1_OFF_FLAGS)
	$(RC_PY) -m experiments.eval.run --config $(PAPER_V1_CONFIG) --returns-csv $(PAPER_V1_RETURNS) --factors-csv $(PAPER_V1_FACTORS) --shrinker oas --out $(PAPER_V1_ABLATE_ROOT)/oas_on
	$(RC_PY) -m experiments.eval.run --config $(PAPER_V1_CONFIG) --returns-csv $(PAPER_V1_RETURNS) --factors-csv $(PAPER_V1_FACTORS) --shrinker rie --out $(PAPER_V1_ABLATE_ROOT)/rie_off $(PAPER_V1_OFF_FLAGS)
	$(RC_PY) -m experiments.eval.run --config $(PAPER_V1_CONFIG) --returns-csv $(PAPER_V1_RETURNS) --factors-csv $(PAPER_V1_FACTORS) --shrinker rie --out $(PAPER_V1_ABLATE_ROOT)/rie_on
	$(RC_PY) tools/make_summary.py --rc-dir $(PAPER_V1_ABLATE_ROOT)
	$(RC_PY) tools/paper_v1_ablation.py --rc-dir $(PAPER_V1_ABLATE_ROOT)

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
	$(RC_VERIFY_DATASET)
	$(RC_VERIFY_FACTORS)
	mkdir -p $(RC_OUT_SANITY) $(RC_LITE_BASE) $(RC_LITE_CACHE)
	$(RC_PY) experiments/eval/run.py \
		--returns-csv $(RC_RETURNS) \
		--window 60 \
		--horizon 10 \
		--start 2023-01-01 \
		--end 2023-06-30 \
		--assets-top 50 \
		--group-design dow \
		--group-min-count $(RC_DOW_GROUP_MIN) \
		--group-min-replicates $(RC_DOW_GROUP_REPS) \
		--min-reps-dow 6 \
		--edge-mode $(DOW_EDGE) \
		--shrinker $(RC_DOW_SHRINKER) \
		--prewhiten $(RC_PREWHITEN) \
		--overlay-delta $(RC_OVERLAY_DELTA) \
		--coarse-candidate $(RC_COARSE_CANDIDATE) \
		--gate-mode $(RC_GATE_MODE) \
		$(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC_MIN) \
		--require-isolated \
		--q-max $(RC_Q_MAX) \
		--mv-gamma $(RC_MV_GAMMA) \
		--mv-box $(RC_MV_BOX) \
		--mv-turnover-bps $(RC_MV_TURNOVER_BPS) \
		--mv-condition-cap $(RC_MV_CONDITION_CAP) \
		--use-factor-prewhiten $(RC_USE_FACTOR_PREWHITEN) \
		--factor-csv $(RC_FACTORS) \
		--out $(RC_DOW_SANITY)
	$(RC_PY) experiments/eval/run.py \
		--returns-csv $(RC_RETURNS) \
		--window 60 \
		--horizon 10 \
		--start 2023-01-01 \
		--end 2023-06-30 \
		--assets-top 50 \
		--group-design vol \
		--group-min-count $(RC_VOL_GROUP_MIN) \
		--group-min-replicates $(RC_VOL_GROUP_REPS) \
		--min-reps-vol 6 \
		--edge-mode $(VOL_EDGE) \
		--shrinker $(RC_VOL_SHRINKER) \
		--prewhiten $(RC_PREWHITEN) \
		--overlay-delta $(RC_OVERLAY_DELTA) \
		--coarse-candidate $(RC_COARSE_CANDIDATE) \
		--gate-mode $(RC_GATE_MODE) \
		$(if $(RC_GATE_ACCEPT_NONISOLATED),--gate-accept-nonisolated,) \
		--gate-delta-calibration $(RC_GATE_CALIB) \
		--gate-delta-frac-min $(RC_GATE_DELTA_FRAC_MIN_VOL) \
		--q-max $(Q_MAX_VOL) \
		--mv-gamma $(RC_MV_GAMMA) \
		--mv-box $(RC_MV_BOX) \
		--mv-turnover-bps $(RC_MV_TURNOVER_BPS) \
		--mv-condition-cap $(RC_MV_CONDITION_CAP) \
		--use-factor-prewhiten $(RC_USE_FACTOR_PREWHITEN) \
		--factor-csv $(RC_FACTORS) \
		--out $(RC_VOL_SANITY)
	$(RC_PY) experiments/equity_panel/run.py \
		--config $(RC_SMOKE_CONFIG) \
		--design dow \
		--estimator dealias \
		--output-dir $(RC_WEEKLY_DOW_OUT) \
		--cache-dir $(RC_LITE_CACHE) \
		--resume \
		--precompute-panel \
		--gating-mode calibrated \
		--gating-calibration $(RC_GATE_CALIB) \
		--edge-mode tyler \
		--prewhiten $(RC_PREWHITEN) \
		--use-factor-prewhiten $(RC_USE_FACTOR_PREWHITEN) \
		--factor-csv $(RC_FACTORS)
	$(RC_PY) experiments/equity_panel/run.py \
		--config $(RC_NESTED_SMOKE_CONFIG) \
		--design nested \
		--estimator dealias \
		--output-dir $(RC_WEEKLY_NESTED_OUT) \
		--cache-dir $(RC_LITE_CACHE) \
		--resume \
		--precompute-panel \
		--gating-mode calibrated \
		--gating-calibration $(RC_GATE_CALIB) \
		--edge-mode tyler \
		--prewhiten $(RC_PREWHITEN) \
		--use-factor-prewhiten $(RC_USE_FACTOR_PREWHITEN) \
		--factor-csv $(RC_FACTORS)
	$(RC_PY) tools/make_summary.py --rc-dir $(RC_OUT_SANITY)
	$(RC_PY) tools/summarize_rc_sanity.py \
		--rc-dir $(RC_OUT_SANITY) \
		--dow-dir $(RC_DOW_SANITY) \
		--vol-dir $(RC_VOL_SANITY) \
		--weekly-dow-dir $(RC_WEEKLY_DOW_OUT) \
		--nested-dir $(RC_WEEKLY_NESTED_OUT)

.PHONY: aws\:rc-lite aws\:rc aws\:sweep-calibration aws\:rc-sensitivity
AWS_ARGS ?=

aws\:%:
	scripts/aws_run.sh $* $(AWS_ARGS)

DOW_EDGE := $(if $(EDGE),$(EDGE),tyler)
VOL_EDGE := $(if $(EDGE),$(EDGE),tyler)
RC_DOW_OUT = $(RC_OUT)/dow-$(DOW_EDGE)
RC_VOL_OUT = $(RC_OUT)/vol-$(VOL_EDGE)
RC_DOW_SANITY := $(RC_OUT_SANITY)/dow-$(DOW_EDGE)
RC_VOL_SANITY := $(RC_OUT_SANITY)/vol-$(VOL_EDGE)
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
	python3 experiments/synthetic_oneway/run.py

run-equity:
	python3 experiments/equity_panel/run.py

.PHONY: run\:equity_smoke
run\:equity_smoke:
	PYTHONPATH=src python3 experiments/equity_panel/run.py \
		--config experiments/equity_panel/config.smoke.yaml \
		--gating-mode fixed \
		--gating-diagnostics \
		--minvar-ridge 0.0001 \
		--minvar-box 0.0,0.1 \
		--minvar-condition-cap 1000000000 \
		--turnover-cost 5

.PHONY: run\:equity_nested_smoke_tiny
run\:equity_nested_smoke_tiny:
	PYTHONPATH=src EXEC_MODE=deterministic python3 experiments/equity_panel/run.py \
		--config experiments/equity_panel/config.nested.smoke.tiny.yaml --gating-diagnostics --exec-mode deterministic

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

.PHONY: gpt-bundle
gpt-bundle:
	@set -eu; \
	if [ -z "$(TICKET)" ]; then echo "TICKET is required: make gpt-bundle TICKET=<ticket> RUN_NAME=<run name>" >&2; exit 1; fi; \
	if [ -z "$(RUN_NAME)" ]; then echo "RUN_NAME is required: make gpt-bundle TICKET=$(TICKET) RUN_NAME=<run name>" >&2; exit 1; fi; \
	run_dir="docs/agent_runs/$(RUN_NAME)"; \
	if [ ! -d "$$run_dir" ]; then echo "Run log $$run_dir is required but missing." >&2; exit 1; fi; \
	required_run_files="PROMPT.md COMMANDS.md RESULTS.md TESTS.md META.md"; \
	missing_run=""; \
	for f in $$required_run_files; do \
		if [ ! -e "$$run_dir/$$f" ]; then missing_run="$$missing_run $$run_dir/$$f"; fi; \
	done; \
	if [ -n "$$missing_run" ]; then echo "Run log missing required files:$$missing_run" >&2; exit 1; fi; \
	required_files="AGENTS.md docs/PLAN_OF_RECORD.md docs/DOCS_AND_LOGGING_SYSTEM.md docs/CODEX_SPRINT_TICKETS.md project_state/CURRENT_RESULTS.md project_state/KNOWN_ISSUES.md project_state/CONFIG_REFERENCE.md PROGRESS.md"; \
	missing=""; \
	for f in $$required_files; do \
		if [ ! -e "$$f" ]; then missing="$$missing $$f"; fi; \
	done; \
	if [ -n "$$missing" ]; then echo "Missing required files:$$missing" >&2; exit 1; fi; \
	tmp=$$(mktemp -d); \
	repo_root=$$(pwd); \
	mkdir -p "$$tmp/docs" "$$tmp/project_state" "$$tmp/docs/agent_runs"; \
	cp AGENTS.md "$$tmp/"; \
	cp PROGRESS.md "$$tmp/"; \
	cp docs/PLAN_OF_RECORD.md "$$tmp/docs/"; \
	cp docs/DOCS_AND_LOGGING_SYSTEM.md "$$tmp/docs/"; \
	cp docs/CODEX_SPRINT_TICKETS.md "$$tmp/docs/"; \
	cp project_state/CURRENT_RESULTS.md "$$tmp/project_state/"; \
	cp project_state/KNOWN_ISSUES.md "$$tmp/project_state/"; \
	cp project_state/CONFIG_REFERENCE.md "$$tmp/project_state/"; \
	cp -r "$$run_dir" "$$tmp/docs/agent_runs/"; \
	diff_rev="$${DIFF_REV:-HEAD}"; \
	python tools/gpt_bundle.py diff --repo "$$repo_root" --rev "$$diff_rev" --output "$$tmp/DIFF.patch"; \
	if [ ! -s "$$tmp/DIFF.patch" ]; then echo "DIFF.patch is empty; aborting bundle." >&2; exit 1; fi; \
	git log -1 --stat > "$$tmp/LAST_COMMIT.txt"; \
	if [ ! -s "$$tmp/LAST_COMMIT.txt" ]; then echo "LAST_COMMIT.txt is empty; aborting bundle." >&2; exit 1; fi; \
	bundle_dir="$$repo_root/docs/gpt_bundles"; \
	mkdir -p "$$bundle_dir"; \
	stamp="$${BUNDLE_STAMP:-$$(date +%Y%m%d_%H%M%S)}"; \
	out="$$bundle_dir/$${stamp}_$(TICKET)_$(RUN_NAME).zip"; \
	(cd "$$tmp" && zip -r "$$out" . >/dev/null); \
	echo "$$out"; \
	rm -rf "$$tmp"
