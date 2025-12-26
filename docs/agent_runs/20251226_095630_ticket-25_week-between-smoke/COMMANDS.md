1. ls
2. cat AGENTS.md
3. cat docs/DOCS_AND_LOGGING_SYSTEM.md
4. git status --porcelain=v1
5. git checkout -b codex/ticket-25_inject-component-modes
6. sed -n '1,240p' experiments/eval/inject_spike.py
7. sed -n '240,520p' experiments/eval/inject_spike.py
8. sed -n '520,1040p' experiments/eval/inject_spike.py
9. rg -n "groups_for_design|group_design" -n src/eval src/experiments -S
10. rg -n "groups_for_design|group_design" src experiments -S
11. sed -n '1,220p' experiments/daily/grouping.py
12. sed -n '220,520p' experiments/daily/grouping.py
13. rg -n "def _build_grouped_window|_required_replicates" experiments/eval/run.py -n
14. sed -n '1280,1365p' experiments/eval/run.py
15. sed -n '1,240p' src/fjs/overlay.py
16. rg -n "diagnostics|off_component" src/fjs/dealias.py -n
17. sed -n '400,520p' src/fjs/dealias.py
18. sed -n '1,240p' tests/experiments/test_inject_spike.py
19. rg -n "_make_injection_basis" -S
20. sed -n '640,820p' experiments/eval/inject_spike.py
21. sed -n '820,1040p' experiments/eval/inject_spike.py
22. sed -n '1040,1240p' experiments/eval/inject_spike.py
23. python -m pytest tests/experiments/test_inject_spike.py -q
24. make test-fast
25. ls reports/fixtures
26. ls experiments/eval
27. date +%Y%m%d_%H%M%S
28. PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,12,24 --inject-mode between --max-windows 20 --window-sampling random --window-sampling-seed 7 --seed 7 --run-id 20251226_095630_ticket-25_week-between-smoke --out reports/inject_spike
29. ls reports/inject_spike/20251226_095630_ticket-25_week-between-smoke
30. python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/curve.csv')
print(path)
print(pd.read_csv(path))
PY
31. python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/windows_detail.csv')
df = pd.read_csv(path)
print(df.columns)
mu24 = df[df['injected_mu']==24.0]
print('mu=24 rows:', len(mu24))
if not mu24.empty:
    print(mu24[['detected_initial','accepted','pre_gate_raw_outliers_found','gating_rejected']].head())
    print('raw_outliers_found>0 share:', (mu24['pre_gate_raw_outliers_found']>0).mean())
PY
32. python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/gating_reasons.csv')
df = pd.read_csv(path)
print(df.head())
mu0 = df[df['injected_mu']==0.0]
if not mu0.empty:
    top = mu0.sort_values('count', ascending=False).head(10)
    print('\nTop reasons mu=0:')
    print(top[['stage','reason','count']])
mu24 = df[df['injected_mu']==24.0]
if not mu24.empty:
    top24 = mu24.sort_values('count', ascending=False).head(10)
    print('\nTop reasons mu=24:')
    print(top24[['stage','reason','count']])
PY
33. ls -a | rg -n "gitignore|ignore" -S
34. cat .gitignore
35. sed -n '1,200p' project_state/CONFIG_REFERENCE.md
36. rg -n "inject_spike" project_state -S
37. RUN_NAME=20251226_095630_ticket-25_week-between-smoke; mkdir -p docs/agent_runs/${RUN_NAME}/artifacts; cp reports/inject_spike/${RUN_NAME}/curve.csv docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/curve.png docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/gating_reasons.csv docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/windows_detail.csv docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/run.json docs/agent_runs/${RUN_NAME}/artifacts/; cp reports/inject_spike/${RUN_NAME}/resolved_config.json docs/agent_runs/${RUN_NAME}/artifacts/
38. cat <<'EOF' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/PROMPT.md
... (see PROMPT.md for full content)
EOF
39. cat <<'EOF' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/COMMANDS.md (failed; EOF not found)
40. cat <<'EOC' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/COMMANDS.md
... (this file)
EOC
41. cat <<'EOF' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/TESTS.md
... (see TESTS.md for full content)
EOF
42. cat <<'EOF' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/RESULTS.md
... (see RESULTS.md for full content)
EOF
43. git rev-parse HEAD
44. sha256sum reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/resolved_config.json
45. python - <<'PY'
import json
from pathlib import Path
path = Path('reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/run.json')
run = json.loads(path.read_text())
print('returns_sha256', run['data'].get('returns_sha256'))
print('factors_sha256', run['data'].get('factors_sha256'))
PY
46. git branch --show-current
47. git status --porcelain=v1
48. python - <<'PY'
import json
from pathlib import Path
path = Path('reports/inject_spike/20251226_095630_ticket-25_week-between-smoke/run.json')
run = json.loads(path.read_text())
print('git_sha', run.get('git_sha'))
print('git_dirty', run.get('git_dirty'))
PY
49. cat <<'EOF' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/META.md
... (see META.md for full content)
EOF
50. date -u +%Y-%m-%dT%H:%MZ
51. cat <<'EOF' >> PROGRESS.md
... (ticket-25 entry)
EOF
52. git add experiments/eval/inject_spike.py tests/experiments/test_inject_spike.py project_state/CONFIG_REFERENCE.md
53. git commit -m "Add inject-mode for spike evaluation" -m "Tests: python -m pytest tests/experiments/test_inject_spike.py -q" -m "Tests: make test-fast"
54. git rev-parse HEAD
55. python - <<'PY'
from pathlib import Path
path = Path('docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/META.md')
text = path.read_text()
text = text.replace('3c347a1b1e502e299f2a5f6943d8923f5bb32d68','3c347a19350e4227a58018dc240e646e720ae68f')
path.write_text(text)
PY
