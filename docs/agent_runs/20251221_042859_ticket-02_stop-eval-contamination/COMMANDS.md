# Commands

git status --porcelain

date +%Y%m%d_%H%M%S

mkdir -p docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/PROMPT.md <<'PROMPT_EOF'
...prompt written...
PROMPT_EOF

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/COMMANDS.md <<'EOF'
# Commands

git status --porcelain

date +%Y%m%d_%H%M%S

mkdir -p docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/PROMPT.md <<'EOF'
...prompt written...
EOF
EOF

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/COMMANDS.md <<'EOF'
# Commands

git status --porcelain

date +%Y%m%d_%H%M%S

mkdir -p docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/PROMPT.md <<'EOF'
...prompt written...
EOF
EOF
EOF

git rev-parse HEAD

git checkout -b feat/ticket-02-stop-eval-contamination

sed -n "1,200p" AGENTS.md

rg -n "ticket-02|Ticket 2|Ticket #2" docs/CODEX_SPRINT_TICKETS.md && sed -n "1,260p" docs/CODEX_SPRINT_TICKETS.md

rg -n "cap|headline|summary|valid" docs/PLAN_OF_RECORD.md && sed -n "1,260p" docs/PLAN_OF_RECORD.md

rg -n "cap_active|cap_sources|max-windows|max_windows|cap" experiments/eval/run.py src/meta/completeness.py tools/make_summary.py

sed -n '2720,3820p' experiments/eval/run.py

sed -n '1,260p' src/meta/completeness.py

rg -n "run_manifest" -S experiments/eval/run.py tools/make_summary.py src

sed -n '520,880p' tools/make_summary.py

sed -n '1,220p' tools/make_summary.py

rg -n "mv_skip_on_missing_solver|skip_on_missing_solver" -S experiments/eval/run.py tools/make_summary.py src tests

ls reports/rc-20251220-sanity-20251220_233700

ls reports/rc-20251220-sanity-20251220_233700/dow-tyler

rg -n "def _write_overlay_toggle" -n experiments/eval/run.py

sed -n '1120,1180p' experiments/eval/run.py

rg -n "SummaryArtifacts" -S

ls reports/rc-test

ls reports/rc-test/dow-tyler-short | head

ls reports/rc-test/dow-tyler-short | rg -n "run.json|resolved_config"

python - <<'PY'
import json
from pathlib import Path
path = Path('reports/rc-20251220-sanity-20251220_233700/dow-tyler/run.json')
with path.open() as f:
    data=json.load(f)
print(list(data.keys())[:5])
print('config' in data, 'execution' in data, 'windows' in data)
print(data['config'].get('mv_skip_on_missing_solver'))
PY

rg -n "cap|contamination|solver" project_state/KNOWN_ISSUES.md && sed -n '1,200p' project_state/KNOWN_ISSUES.md

sed -n '1,200p' PROGRESS.md

rg -n "add_argument\\(.*returns" -n experiments/eval/run.py

rg -n "returns_csv|returns-csv" experiments/eval/run.py

sed -n '620,940p' experiments/eval/run.py

rg -n "resolve_eval_config" -S experiments

sed -n '1,220p' experiments/eval/config.py

ls data

python - <<'PY'
import json
from pathlib import Path
for run in Path('reports/rc-test').iterdir():
    if run.is_dir():
        path = run / 'run.json'
        if path.exists():
            data = json.loads(path.read_text())
            cap = data.get('windows', {}).get('cap_active')
            print(run.name, cap, data.get('windows', {}).get('cap_sources'))
PY

python - <<'PY'
import json
from pathlib import Path
path = Path('reports/rc-test/dow-tyler-short/run.json')
print(path.exists())
data = json.loads(path.read_text())
print(data.keys())
print(data.get('windows'))
PY

cat docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/COMMANDS.md

make test-fast

EXEC_MODE=deterministic make rc-lite-sanity
EXEC_MODE=deterministic make rc-lite-sanity
EXEC_MODE=deterministic make rc-lite-sanity

EXEC_MODE=deterministic PYTHONPATH=src:. OMP_NUM_THREADS=1 python3 experiments/eval/run.py \
	--returns-csv data/returns_daily.csv \
	--window 60 \
	--horizon 10 \
	--start 2023-01-01 \
	--end 2023-06-30 \
	--assets-top 50 \
	--group-design vol \
	--group-min-count 3 \
	--group-min-replicates 6 \
	--min-reps-vol 6 \
	--edge-mode tyler \
	--shrinker oas \
	--prewhiten ff5mom \
	--overlay-delta 0.05 \
	--coarse-candidate 1 \
	--gate-mode soft \
	--gate-accept-nonisolated \
	--gate-delta-calibration calibration/edge_delta_thresholds.json \
	--gate-delta-frac-min 0.015 \
	--q-max 2 \
	--mv-gamma 1e-4 \
	--mv-box 0.0,0.1 \
	--mv-turnover-bps 5 \
	--mv-condition-cap 1000000 \
	--use-factor-prewhiten 1 \
	--factor-csv data/factors/ff5mom_daily.csv \
	--out reports/rc-20251221-sanity-20251221_045550/vol-tyler

ls reports/rc-20251221-sanity-20251221_045550

PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251221-sanity-20251221_045550

EXEC_MODE=deterministic python -m experiments.eval.run --returns-csv data/returns_daily.csv --window 40 --horizon 5 --out reports/smoke_cap_test --assets-top 20 --group-design dow --shrinker rie --prewhiten off --use-factor-prewhiten 0 --q-max 2 --mv-box-lo -0.25 --mv-box-hi 0.25 --mv-turnover-bps 0.0 --mv-condition-cap 1000000 --max-windows 5 --min-comparison-windows 3 --seed 123 --workers 1

PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/smoke_cap_test

sed -n '1,160p' reports/smoke_cap_test/summary/limitations.md

python - <<'PY'
import pandas as pd
from pathlib import Path
perf = pd.read_csv(Path('reports/smoke_cap_test/summary/summary_perf.csv'))
det = pd.read_csv(Path('reports/smoke_cap_test/summary/summary_detection.csv'))
print('perf rows', perf.shape)
print('det rows', det.shape)
print(perf.head(5).to_string(index=False))
PY

rg -n "rc-lite-sanity" Makefile

sed -n '178,260p' Makefile

git status --porcelain

sed -n '1,160p' reports/rc-20251221-sanity-20251221_045550/summary/limitations.md

python - <<'PY'
import pandas as pd
from pathlib import Path
perf = pd.read_csv(Path('reports/rc-20251221-sanity-20251221_045550/summary/summary_perf.csv'))
det = pd.read_csv(Path('reports/rc-20251221-sanity-20251221_045550/summary/summary_detection.csv'))
print(perf.shape, det.shape)
PY

date -u +
date -u +%Y-%m-%dT%H:%MZ

printf '\ndate -u +%Y-%m-%dT%H:%MZ\n' >> docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/COMMANDS.md

python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
entry = reg.get('returns_daily.csv')
print('returns_daily.csv', entry)
fac = json.loads(Path('data/factors/registry.json').read_text())
fac_entry = fac.get('ff5mom_daily.csv')
print('ff5mom_daily.csv', fac_entry)
PY

python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
print(list(reg.keys())[:10])
PY

python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
datasets = reg.get('datasets', {})
print(list(datasets.keys())[:10])
print(datasets.get('returns_daily.csv'))
PY

python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
entry = reg.get('datasets', {}).get('data/returns_daily.csv')
print(entry)
fac = json.loads(Path('data/factors/registry.json').read_text())
fac_entry = fac.get('datasets', {}).get('data/factors/ff5mom_daily.csv')
print(fac_entry)
PY

python - <<'PY'
import hashlib
from pathlib import Path
paths = [
    Path('experiments/eval/config.yaml'),
    Path('experiments/eval/thresholds.json'),
    Path('calibration/edge_delta_thresholds.json'),
]
for path in paths:
    data = path.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    print(path, h)
PY

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/TESTS.md <<'EOF'
...updated TESTS.md...
EOF

printf '\ncat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/TESTS.md <<\'EOF\'\n...\nEOF\n' >> docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/COMMANDS.md

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/META.md <<'EOF'
...updated META.md...
EOF

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/RESULTS.md <<'EOF'
...updated RESULTS.md...
EOF

git add experiments/eval/run.py tools/make_summary.py src/meta/completeness.py tests/tools/test_make_summary.py

git commit -m "Guard summary eligibility against caps and solver skips" -m "Tests run: make test-fast"

git rev-parse 6df53dd

cat > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/META.md <<'EOF'
...updated META.md with code commit...
EOF

git add PROGRESS.md docs/CODEX_SPRINT_TICKETS.md project_state/KNOWN_ISSUES.md docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination

git diff eb880be520eab2f58d50c2efed8b55385ce93488..HEAD > docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/DIFF.patch

git add docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/DIFF.patch docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/COMMANDS.md

make gpt-bundle TICKET=ticket-02 RUN_NAME=20251221_042859_ticket-02_stop-eval-contamination

cat >> docs/agent_runs/20251221_042859_ticket-02_stop-eval-contamination/RESULTS.md <<'EOF'
...added bundle path...
EOF

git status --porcelain
