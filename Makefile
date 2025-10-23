.PHONY: setup fmt lint test run-synth run-equity

setup:
	pip install --upgrade pip
	pip install -e '.[dev]'

fmt:
	black src experiments tests
	ruff check --fix src experiments tests

lint:
	ruff check src experiments tests
	mypy src

test:
	pytest -q

.PHONY: test-fast test-all
test-fast:
	pytest -q -m "not slow" -n auto --maxfail=1 || pytest -q -m "not slow" --maxfail=1

test-all:
	pytest -q -n auto || pytest -q

.PHONY: test-progress
test-progress:
	# Verbose output with progress bar (pytest-sugar), parallel if available
	pytest -v -n auto || pytest -v

run-synth:
	python experiments/synthetic_oneway/run.py

run-equity:
	python experiments/equity_panel/run.py

.PHONY: figures
figures: ## regenerate all figures (synthetic + equity)
	python experiments/synthetic_oneway/run.py
	python experiments/equity_panel/run.py
