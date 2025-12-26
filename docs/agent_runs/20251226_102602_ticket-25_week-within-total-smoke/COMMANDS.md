1. git status --porcelain=v1
2. ls -l docs/agent_runs/20251226_095630_ticket-25_week-between-smoke
3. cat <<'EOF' > docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/TESTS.md
... (see TESTS.md for full content)
4. date +%Y%m%d_%H%M%S
5. python -m pytest tests/experiments/test_inject_spike.py -q
6. make test-fast
7. PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,12,24 --inject-mode within --max-windows 20 --window-sampling random --window-sampling-seed 7 --seed 7 --run-id 20251226_102602_ticket-25_week-within-smoke --out reports/inject_spike
8. PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv reports/fixtures/returns_daily_small.csv --factors-csv reports/fixtures/ff5mom_daily_small.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 0,12,24 --inject-mode total --max-windows 20 --window-sampling random --window-sampling-seed 7 --seed 7 --run-id 20251226_102602_ticket-25_week-total-smoke --out reports/inject_spike
9. python - <<'PY'
import pandas as pd
from pathlib import Path

runs = {
    'within': Path('reports/inject_spike/20251226_102602_ticket-25_week-within-smoke'),
    'total': Path('reports/inject_spike/20251226_102602_ticket-25_week-total-smoke'),
}

for mode, root in runs.items():
    print(f"== {mode} ==")
    curve = pd.read_csv(root / 'curve.csv')
    print(curve)
    detail = pd.read_csv(root / 'windows_detail.csv')
    mu24 = detail[detail['injected_mu'] == 24.0]
    if mu24.empty:
        print('mu=24 windows: 0')
    else:
        print('mu=24 windows:', len(mu24))
        print('raw_outliers_found>0 share:', (mu24['pre_gate_raw_outliers_found']>0).mean())
    gating = pd.read_csv(root / 'gating_reasons.csv')
    for mu in [0.0, 24.0]:
        subset = gating[gating['injected_mu'] == mu]
        if subset.empty:
            print(f'mu={mu} reasons: none')
            continue
        top = subset.sort_values('count', ascending=False).head(5)
        print(f'mu={mu} top reasons:')
        print(top[['stage','reason','count']])
    print()
PY
10. RUN_NAME=20251226_102602_ticket-25_week-within-total-smoke; mkdir -p docs/agent_runs/${RUN_NAME}/artifacts/within docs/agent_runs/${RUN_NAME}/artifacts/total; cp reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/curve.csv docs/agent_runs/${RUN_NAME}/artifacts/within/; cp reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/curve.png docs/agent_runs/${RUN_NAME}/artifacts/within/; cp reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/gating_reasons.csv docs/agent_runs/${RUN_NAME}/artifacts/within/; cp reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/windows_detail.csv docs/agent_runs/${RUN_NAME}/artifacts/within/; cp reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/run.json docs/agent_runs/${RUN_NAME}/artifacts/within/; cp reports/inject_spike/20251226_102602_ticket-25_week-within-smoke/resolved_config.json docs/agent_runs/${RUN_NAME}/artifacts/within/; cp reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/curve.csv docs/agent_runs/${RUN_NAME}/artifacts/total/; cp reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/curve.png docs/agent_runs/${RUN_NAME}/artifacts/total/; cp reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/gating_reasons.csv docs/agent_runs/${RUN_NAME}/artifacts/total/; cp reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/windows_detail.csv docs/agent_runs/${RUN_NAME}/artifacts/total/; cp reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/run.json docs/agent_runs/${RUN_NAME}/artifacts/total/; cp reports/inject_spike/20251226_102602_ticket-25_week-total-smoke/resolved_config.json docs/agent_runs/${RUN_NAME}/artifacts/total/
11. cat <<'EOF' > docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/PROMPT.md
... (see PROMPT.md for full content)
12. cat <<'EOF' > docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/TESTS.md
... (see TESTS.md for full content)
13. cat <<'EOF' > docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/RESULTS.md
... (see RESULTS.md for full content)
14. python - <<'PY'
import json
from pathlib import Path

def summarize(root):
    run = json.loads((root / 'run.json').read_text())
    return {
        'git_sha': run.get('git_sha'),
        'git_dirty': run.get('git_dirty'),
        'returns_sha256': run.get('data', {}).get('returns_sha256'),
        'factors_sha256': run.get('data', {}).get('factors_sha256'),
    }

roots = {
    'within': Path('reports/inject_spike/20251226_102602_ticket-25_week-within-smoke'),
    'total': Path('reports/inject_spike/20251226_102602_ticket-25_week-total-smoke'),
}
for mode, root in roots.items():
    summary = summarize(root)
    print(mode, summary)
    cfg = root / 'resolved_config.json'
    import hashlib
    data = cfg.read_bytes()
    print(mode, 'resolved_config_sha256', hashlib.sha256(data).hexdigest())
PY
15. git rev-parse HEAD
16. cat <<'EOF' > docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/META.md
... (see META.md for full content)
17. python - <<'PY'
from pathlib import Path
path = Path('docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/META.md')
text = path.read_text()
text = text.replace('git_sha_end: TBD','git_sha_end: e198fddf2cb74df8e41b5a9d043e45f300115aae')
text = text.replace('git_dirty_end: TBD','git_dirty_end: true')
path.write_text(text)
PY
18. date +%Y%m%d_%H%M%S
19. python - <<'PY'
from pathlib import Path
path = Path('docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/RESULTS.md')
text = path.read_text()
bundle = 'docs/gpt_bundles/20251226_103353_ticket-25_20251226_102602_ticket-25_week-within-total-smoke.zip'
lines = text.splitlines()
out = []
for line in lines:
    if line.strip().startswith('- Pending'):
        out.append(f'- {bundle}')
    else:
        out.append(line)
path.write_text('\n'.join(out) + '\n')
PY
20. cat <<'EOF' >> PROGRESS.md
... (ticket-25 within/total entry)
21. cat <<'END_COMMANDS_LOG' > docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke/COMMANDS.md
... (this file)
22. git add docs/agent_runs/20251226_095630_ticket-25_week-between-smoke/TESTS.md docs/agent_runs/20251226_102602_ticket-25_week-within-total-smoke PROGRESS.md
23. git commit -m "Add within/total smoke run log" -m "Tests: python -m pytest tests/experiments/test_inject_spike.py -q" -m "Tests: make test-fast"
24. BUNDLE_STAMP=20251226_103353 make gpt-bundle TICKET=ticket-25 RUN_NAME=20251226_102602_ticket-25_week-within-total-smoke
