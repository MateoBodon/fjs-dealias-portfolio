1. date +%Y%m%d_%H%M%S
2. python -m pytest tests/experiments/test_inject_spike.py -q
3. make test-fast
4. PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,6,12,18,24,30,36 --inject-mode between --inject-frac-min 0.2 --inject-frac-max 0.2 --max-windows 48 --window-sampling first --window-sampling-seed 7 --seed 7 --run-id 20251226_105628_ticket-25_week-between-stress --out reports/inject_spike
5. python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path('reports/inject_spike/20251226_105628_ticket-25_week-between-stress')
curve = pd.read_csv(root / 'curve.csv')
print(curve)
window = pd.read_csv(root / 'windows_detail.csv')
mu36 = window[window['injected_mu'] == 36.0]
print('mu=36 windows', len(mu36))
if not mu36.empty:
    print('raw_outliers_found>0 share', (mu36['pre_gate_raw_outliers_found']>0).mean())

gating = pd.read_csv(root / 'gating_reasons.csv')
for mu in [0.0, 36.0]:
    subset = gating[gating['injected_mu'] == mu]
    if subset.empty:
        print('mu', mu, 'no reasons')
        continue
    top = subset.sort_values('count', ascending=False).head(6)
    print('mu', mu, 'top reasons')
    print(top[['stage','reason','count']])
PY
6. RUN_NAME=20251226_105628_ticket-25_week-between-stress; mkdir -p docs/agent_runs/${RUN_NAME}/artifacts; cp reports/inject_spike/${RUN_NAME}/curve.csv docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/curve.png docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/gating_reasons.csv docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/windows_detail.csv docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/run.json docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/resolved_config.json docs/agent_runs/${RUN_NAME}/artifacts/
7. cat <<'EOF' > docs/agent_runs/20251226_105628_ticket-25_week-between-stress/PROMPT.md
... (see PROMPT.md for full content)
8. cat <<'EOF' > docs/agent_runs/20251226_105628_ticket-25_week-between-stress/TESTS.md
... (see TESTS.md for full content)
9. cat <<'EOF' > docs/agent_runs/20251226_105628_ticket-25_week-between-stress/RESULTS.md
... (see RESULTS.md for full content)
10. python - <<'PY'
import json
from pathlib import Path
run = json.loads(Path('reports/inject_spike/20251226_105628_ticket-25_week-between-stress/run.json').read_text())
print('git_sha', run.get('git_sha'))
print('git_dirty', run.get('git_dirty'))
print('returns_sha256', run.get('data', {}).get('returns_sha256'))
print('factors_sha256', run.get('data', {}).get('factors_sha256'))
PY
11. sha256sum reports/inject_spike/20251226_105628_ticket-25_week-between-stress/resolved_config.json
12. git rev-parse HEAD
13. cat <<'EOF' > docs/agent_runs/20251226_105628_ticket-25_week-between-stress/META.md
... (see META.md for full content)
14. cat <<'EOF' >> PROGRESS.md
... (ticket-25 between stress entry)
15. cat <<'EOF' > docs/agent_runs/20251226_105628_ticket-25_week-between-stress/COMMANDS.md
... (this file)
16. git add docs/agent_runs/20251226_105628_ticket-25_week-between-stress PROGRESS.md
17. git commit -m "Add between stress-test run log" -m "Tests: python -m pytest tests/experiments/test_inject_spike.py -q" -m "Tests: make test-fast"
18. BUNDLE_STAMP=20251226_1112xx make gpt-bundle TICKET=ticket-25 RUN_NAME=20251226_105628_ticket-25_week-between-stress
