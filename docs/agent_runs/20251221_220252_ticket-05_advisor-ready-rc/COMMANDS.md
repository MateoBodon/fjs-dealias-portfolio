# Commands (ticket-05)

# Pre-log commands (ran before COMMANDS.md existed; recorded verbatim)
- git checkout main
- date +%Y%m%d_%H%M%S
- git stash push -u -m "local pre ticket-05"
- git status -sb
- git pull
- git push
- git checkout -b feat/ticket-05-advisor-ready-rc

# Commands (logged live below)
- RUN_NAME=20251221_220252_ticket-05_advisor-ready-rc RUN_DIR=docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc mkdir -p "docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc" && create PROMPT.md/COMMANDS.md/RESULTS.md/TESTS.md/META.md
- make test-fast
- EXEC_MODE=deterministic make rc-lite-sanity
- PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251221-sanity-20251221_220443
- wc -l reports/rc-20251221-sanity-20251221_220443/summary/overlay_forensics.csv
- cat reports/rc-20251221-sanity-20251221_220443/summary/limitations.md
- RUN_TS=20251221_221851 RUN_DIR=reports/rc-ticket-05-<ts> mkdir -p "docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc" && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --out "docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/dow-paper-v1" (failed: missing --returns-csv)
- ps -eo pid,etime,cmd | rg "experiments/eval/run.py"
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- ps -o pid,etime,pcpu,pmem,cmd -p 2143641
- RUN_TS=20251221_221902 RUN_DIR=reports/rc-ticket-05-20251221_221902 mkdir -p "docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc" && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out "docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/dow-paper-v1"
- PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-05-20251221_221902
- rg -n "rc-dow" Makefile
- sed -n '330,380p' Makefile
- rg -n "RC_START|RC_END" Makefile
- cat experiments/eval/config.paper_v1.yaml
- PYTHONPATH=src:. python experiments/eval/run.py --help | head -n 40
- ls -la reports/rc-ticket-05-20251221_221902/dow-paper-v1
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'
with open(p) as f:
    d=json.load(f)
print('cap_active', d.get('cap_active'))
print('cap_sources', d.get('cap_sources'))
PY
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'
with open(p) as f:
    d=json.load(f)
print([k for k in d.keys() if 'cap' in k])
PY
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'
with open(p) as f:
    d=json.load(f)
for k in ['config','limits','run_meta','data_meta','validity']:
    if k in d:
        print(k, type(d[k]))
print('keys', list(d.keys()))
PY
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'
with open(p) as f:
    d=json.load(f)
print(d.get('outputs'))
PY
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/summary/completeness.json'
with open(p) as f:
    d=json.load(f)
print(d)
PY
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'
with open(p) as f:
    d=json.load(f)
print({k: d['config'].get(k) for k in ['start','end','max_windows'] if k in d['config']})
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/summary/summary_perf.csv'
df=pd.read_csv(p)
print(df)
PY
- python - <<'PY'
import pandas as pd
for name in ['summary_detection','summary_skip_stats','summary_perf']:
    p=f'reports/rc-ticket-05-20251221_221902/summary/{name}.csv'
    df=pd.read_csv(p)
    print(name, df.shape)
PY
- rg -n "run.json|rc_run|summary_perf" tools/make_summary.py
- sed -n '150,240p' tools/make_summary.py
- rg -n "use_design_dirs" tools/make_summary.py
- sed -n '640,730p' tools/make_summary.py
- rg -n "evaluate_eval_run" -g"*.py"
- sed -n '1,220p' src/meta/completeness.py
- python - <<'PY'
import json
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'
with open(p) as f:
    d=json.load(f)
print('mv_skip_on_missing_solver', d['config'].get('mv_skip_on_missing_solver'))
PY
- ls -la reports/rc-ticket-05-20251221_221902/summary
- cat reports/rc-ticket-05-20251221_221902/summary/limitations.md
- ls -la reports/rc-ticket-05-20251221_221902/dow-paper-v1/full | head -n 20
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
df=pd.read_csv(p)
print(df.columns.tolist())
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
df=pd.read_csv(p)
print('rows', len(df))
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
df=pd.read_csv(p)
print('design_ok false', (df['design_ok']==0).sum())
print('design_ok true', (df['design_ok']==1).sum())
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
row=pd.read_csv(p).loc[lambda d: d['design_ok']==0].iloc[0]
print(row)
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
df=pd.read_csv(p)
print('window_start NaN', df['window_start'].isna().sum())
print('design_ok false & window_start NaN', df[df['design_ok']==0]['window_start'].isna().sum())
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
fail=pd.read_csv(p).loc[lambda d: d['design_ok']==0]
print('min group_count', fail['group_count'].min(), 'min required', fail['group_count_required'].min())
print('min group_replicates', fail['group_replicates'].min(), 'required', fail['group_replicates_required'].min())
print('min reps_by_label', fail['reps_by_label'].head().tolist())
print('unique group_replicates_required', fail['group_replicates_required'].unique()[:5])
print('unique group_count_required', fail['group_count_required'].unique()[:5])
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
fail=pd.read_csv(p).loc[lambda d: d['design_ok']==0]
print(fail[['window_start','group_count','group_replicates','group_count_required','group_replicates_required','group_label_counts','reps_by_label','design_ok']].head(3))
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
fail=pd.read_csv(p).loc[lambda d: d['design_ok']==0]
print(fail['reason_code'].value_counts(dropna=False).head())
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
fail=pd.read_csv(p).loc[lambda d: d['design_ok']==0]
print(fail['factor_present'].value_counts(dropna=False).head())
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
fail=pd.read_csv(p).loc[lambda d: d['design_ok']==0]
print(fail.isna().mean().sort_values(ascending=False).head(10))
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'
fail=pd.read_csv(p).loc[lambda d: d['design_ok']==0]
print('group_count unique', fail['group_count'].unique()[:5])
print('group_replicates unique', fail['group_replicates'].unique()[:5])
print('group_observations unique', fail['group_observations'].unique()[:5])
PY
- python - <<'PY'
import pandas as pd
p='reports/rc-ticket-05-20251221_221902/dow-paper-v1/skip_stats.csv'
print(pd.read_csv(p))
PY
- rg -n "design_ok"
- sed -n '2600,2760p' experiments/eval/run.py
- sed -n '1200,1260p' experiments/eval/run.py
- rg -n "group_count" experiments/eval/run.py | head -n 20
- sed -n '1780,2140p' experiments/eval/run.py
- rg -n "holdout" experiments/eval/run.py src | head -n 20
- rg -n "missing" experiments/eval/run.py | head -n 40
- python - <<'PY'
import pandas as pd
rets=pd.read_csv('data/returns_daily.csv', nrows=5)
print(rets.columns.tolist())
print(rets.head())
PY
- python - <<'PY'
import pandas as pd
rets=pd.read_csv('data/returns_daily.csv')
counts=rets.groupby('date')['asset'].nunique()
print('min assets per date', counts.min())
print('dates with <60 assets', (counts<60).sum())
print('first date with <60', counts[counts<60].head())
PY  # failed: Column not found: asset
- python - <<'PY'
import pandas as pd
rets=pd.read_csv('data/returns_daily.csv')
counts=rets.groupby('date')['ticker'].nunique()
print('min assets per date', counts.min())
print('dates with <60 assets', (counts<60).sum())
print('sample dates with <60', counts[counts<60].head())
PY
- python - <<'PY'
import pandas as pd
rets=pd.read_csv('data/returns_daily.csv', usecols=['date'])
factors=pd.read_csv('data/factors/ff5mom_daily.csv', usecols=['date'])
missing=set(rets['date'].unique())-set(factors['date'].unique())
print('missing factor dates', len(missing))
print('first missing', list(sorted(missing))[:5])
PY
- python - <<'PY'
import pandas as pd
rets=pd.read_csv('data/returns_daily.csv', usecols=['date'])
print('unique dates', rets['date'].nunique())
PY
- python - <<'PY'
import pandas as pd
rets=pd.read_csv('data/returns_daily.csv')
factors=pd.read_csv('data/factors/ff5mom_daily.csv')
print('returns dates', rets['date'].min(), rets['date'].max(), len(rets))
print('factors dates', factors['date'].min(), factors['date'].max(), len(factors))
PY
- wc -l reports/rc-ticket-05-20251221_221902/summary/overlay_forensics.csv
- python - <<'PY'\nimport json\np='reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json'\nwith open(p) as f:\n    d=json.load(f)\nprint(d.get('windows'))\nPY
- python - <<'PY'\nimport pandas as pd\np='reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv'\ncounts=pd.read_csv(p)['reason_code'].value_counts(dropna=False)\nprint(counts.head())\nPY
- git rev-parse HEAD
- sha256sum experiments/eval/config.paper_v1.yaml data/returns_daily.csv data/factors/ff5mom_daily.csv
- make gpt-bundle TICKET=ticket-05 RUN_NAME=20251221_220252_ticket-05_advisor-ready-rc
- git status -sb
- cat >> .git/info/exclude (add docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/ and reports/rc-20251221-sanity-20251221_220443/)
- echo 'reports/rc-20251221-sanity-20251221_220443' >> .git/info/exclude
- cat >> .gitignore (local-only ignores for ticket-05)
- git update-index --skip-worktree .gitignore
- git status -sb
- append follow-up note to docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/RESULTS.md
- append checklist to docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/RESULTS.md
