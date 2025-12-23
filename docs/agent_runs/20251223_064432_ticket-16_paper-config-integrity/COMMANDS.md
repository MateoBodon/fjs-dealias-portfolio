- ls
- rg -n "paper" Makefile experiments -g"*.py" -g"*.yaml"
- sed -n '1,220p' experiments/eval/config.py
- sed -n '220,520p' experiments/eval/config.py
- sed -n '1,240p' experiments/eval/run.py
- rg -n "run\.json|resolved_config|config_hash|git" experiments/eval -g"*.py"
- sed -n '1580,1760p' experiments/eval/run.py
- sed -n '3080,3225p' experiments/eval/run.py
- sed -n '3800,3955p' experiments/eval/run.py
- rg -n "def _write_run_metadata|run_metadata" -n experiments/eval/run.py
- sed -n '392,470p' experiments/eval/run.py
- rg -n "def parse_args|config" experiments/eval/run.py
- ls experiments/eval
- cat experiments/eval/config.paper_v1.yaml
- sed -n '90,210p' Makefile
- rg -n "paper-v1|paper v1|paper_v1|paper config" docs project_state
- sed -n '1,200p' docs/DOCS_AND_LOGGING_SYSTEM.md
- date +%Y%m%d_%H%M%S
- git status --short
- git checkout -b codex/ticket-16-paper-config-integrity
- RUN_NAME=20251223_064432_ticket-16_paper-config-integrity; mkdir -p docs/agent_runs/$RUN_NAME
- cat > docs/agent_runs/20251223_064432_ticket-16_paper-config-integrity/PROMPT.md <<'EOF' ... EOF
- git diff --stat
- apply_patch (experiments/eval/config.py)
- apply_patch (experiments/eval/run.py)
- apply_patch (experiments/eval/run.py)
- apply_patch (tests/experiments/test_eval_run.py)
- apply_patch (project_state/KNOWN_ISSUES.md)
- apply_patch (docs/PLAN_OF_RECORD.md)
- make test-fast
- EXEC_MODE=deterministic make rc-lite-sanity
- python - <<'PY'
import json
from pathlib import Path
run_path = Path('reports/rc-20251223-sanity-20251223_064808/dow-tyler/run.json')
print(run_path.exists())
if run_path.exists():
    data = json.loads(run_path.read_text())
    keys = {k: data.get(k) for k in ['git_sha','git_dirty','resolved_config_path','resolved_config_hash']}
    print(keys)
PY
- python tools/verify_dataset.py data/returns_daily.csv --registry data/registry.json
- python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
entry = reg.get('returns_daily.csv') or reg.get('returns_daily') or next((v for k,v in reg.items() if 'returns_daily' in k), None)
print(entry)
PY
- python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
print(list(reg.keys())[:20])
PY
- python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/registry.json').read_text())
print(type(reg.get('datasets')))
print(reg.get('datasets'))
PY
- python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path('data/factors/registry.json').read_text())
print(reg.get('datasets'))
PY
- python - <<'PY'
import hashlib
from pathlib import Path

paths = [
    Path('reports/rc-20251223-sanity-20251223_064808/dow-tyler/resolved_config.json'),
    Path('reports/rc-20251223-sanity-20251223_064808/vol-tyler/resolved_config.json'),
]
for path in paths:
    data = path.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    print(path, h)
PY
- date -u +%Y-%m-%dT%H:%MZ
