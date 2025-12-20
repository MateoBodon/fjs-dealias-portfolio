- date -u +"%Y%m%d_%H%M%S"
- git status --short --branch
- git checkout -b ticket-10-nested-null-fpr
- sed -n '1,200p' AGENTS.md
- sed -n '1,200p' docs/PLAN_OF_RECORD.md
- rg "ticket-10" -n docs/CODEX_SPRINT_TICKETS.md
- sed -n '30,130p' docs/CODEX_SPRINT_TICKETS.md
- sed -n '1,200p' project_state/KNOWN_ISSUES.md
- export RUN_NAME=20251220_011519_ticket-10_nested-null-fpr && mkdir -p docs/agent_runs/$RUN_NAME
- export RUN_NAME=20251220_011519_ticket-10_nested-null-fpr && cat <<'PROMPT' > docs/agent_runs/$RUN_NAME/PROMPT.md
  (full prompt text pasted)
PROMPT
- export RUN_NAME=20251220_011519_ticket-10_nested-null-fpr && printf "" > docs/agent_runs/$RUN_NAME/COMMANDS.md && printf "" > docs/agent_runs/$RUN_NAME/RESULTS.md && printf "" > docs/agent_runs/$RUN_NAME/TESTS.md && printf "" > docs/agent_runs/$RUN_NAME/META.md
- sed -n '1,160p' docs/DOCS_AND_LOGGING_SYSTEM.md
- cat docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
- sed -n '1,200p' experiments/synthetic/nested_killtest.py
- sed -n '200,400p' experiments/synthetic/nested_killtest.py
- sed -n '400,800p' experiments/synthetic/nested_killtest.py
- cat experiments/synthetic/config.nested.killtest.yaml
- cat calibration/edge_delta_thresholds.json
- python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr
- python3 -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr
- make setup
- python3 -m venv .venv
- source .venv/bin/activate && pip install --upgrade pip && pip install -e '.[dev]'
- source .venv/bin/activate && python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr
- cat reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/summary.csv
- .venv/bin/python -m json.tool reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/run.json
- head -n 5 reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/nested_killtest_trials.csv
- .venv/bin/python - <<'PY'  # isolated_spikes describe
- .venv/bin/python - <<'PY'  # scenario uniques with keep_default_na=False
- .venv/bin/python - <<'PY'  # delta_frac sweep (timed out after 120s)
- .venv/bin/python - <<'PY'  # delta_frac sweep 0.05-0.5 with 10 trials
- .venv/bin/python - <<'PY'  # delta sweep 0.35-2.0 with 10 trials
- .venv/bin/python - <<'PY'  # delta sweep 0.0-0.35 with 10 trials
- .venv/bin/python - <<'PY'  # inspect detections and pre_outlier_count
- .venv/bin/python - <<'PY'  # filtered detections requiring admissible_root
- ls experiments/equity_panel | grep nested
- sed -n '1,200p' experiments/equity_panel/config.nested.smoke.yaml
- rg "calibration_path" -n experiments/equity_panel/run.py
- sed -n '1380,1425p' experiments/equity_panel/run.py
- sed -n '600,760p' experiments/equity_panel/run.py
- rg "design_mode" -n experiments/equity_panel/run.py | head
- sed -n '1200,1280p' experiments/equity_panel/run.py
- sed -n '1,120p' experiments/equity_panel/config.nested.crisis.2020.yaml
- source .venv/bin/activate && python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr --calibration-out calibration/nested_edge_delta_thresholds.json --run-name 20251220_011519_ticket-10_nested-null-fpr --target-fpr 0.02
- cat reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/summary.csv | head
- source .venv/bin/activate && python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr --calibration-out calibration/nested_edge_delta_thresholds.json --run-name 20251220_011519_ticket-10_nested-null-fpr --target-fpr 0.02
- cat reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/summary.csv
- source .venv/bin/activate && python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr --calibration-out calibration/nested_edge_delta_thresholds.json --run-name 20251220_011519_ticket-10_nested-null-fpr --target-fpr 0.02
- cat reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/summary.csv
- ls calibration | grep nested
- head -n 80 calibration/nested_edge_delta_thresholds.json
- cat experiments/synthetic/config.nested.killtest.yaml
- .venv/bin/python -m json.tool reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/run.json
- git status --short
- source .venv/bin/activate && make test-fast
- source .venv/bin/activate && EXEC_MODE=deterministic make run:equity_smoke
- git rev-parse HEAD
- .venv/bin/python - <<'PY'
import json
with open('reports/synthetic/nested_killtest/20251220_011519_ticket-10_nested-null-fpr/run.json') as fh:
    meta = json.load(fh)
print(meta['diagnostics']['tyler'].get('gating_rejections'))
PY
- source .venv/bin/activate && make gpt-bundle TICKET=ticket-10 RUN_NAME=20251220_011519_ticket-10_nested-null-fpr
- git add experiments/synthetic/nested_killtest.py experiments/synthetic/config.nested.killtest.yaml src/fjs/gating.py tests/test_calibration_lookup.py experiments/equity_panel/run.py experiments/equity_panel/config.nested.smoke.yaml experiments/equity_panel/config.nested.crisis.2020.yaml calibration/nested_edge_delta_thresholds.json reports/synthetic/nested_killtest docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr
- git add docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/RESULTS.md
- git add docs/agent_runs/20251220_011519_ticket-10_nested-null-fpr/COMMANDS.md
- git rev-parse HEAD
