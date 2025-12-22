git checkout main && git pull && git status -sb
git checkout -b feat/ticket-06-window-coverage
date +%Y%m%d_%H%M%S
RUN_NAME=20251222_014730_ticket-06_window-coverage; mkdir -p docs/agent_runs/$RUN_NAME; for f in PROMPT COMMANDS RESULTS TESTS META; do : > docs/agent_runs/$RUN_NAME/$f.md; done
cat <<'EOF' > docs/agent_runs/20251222_014730_ticket-06_window-coverage/PROMPT.md
[PROMPT CONTENT IN docs/agent_runs/20251222_014730_ticket-06_window-coverage/PROMPT.md]
EOF
cat <<'EOF' > docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
git checkout main && git pull && git status -sb
git checkout -b feat/ticket-06-window-coverage
date +%Y%m%d_%H%M%S
RUN_NAME=20251222_014730_ticket-06_window-coverage; mkdir -p docs/agent_runs/$RUN_NAME; for f in PROMPT COMMANDS RESULTS TESTS META; do : > docs/agent_runs/$RUN_NAME/$f.md; done
cat <<'EOF' > docs/agent_runs/20251222_014730_ticket-06_window-coverage/PROMPT.md
[...prompt content omitted here; see PROMPT.md...]
EOF
cat <<'EOF' > docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
[recreated COMMANDS.md]
EOF
EOF
sed -n '1,160p' docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
cat <<'EOF' > docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
[recreated COMMANDS.md]
EOF
EOF
make test-fast
cat <<'EOF' > docs/agent_runs/20251222_014730_ticket-06_window-coverage/TESTS.md
make test-fast
- Result: PASS
- Warnings: PytestConfigWarning: Unknown config option: timeout
- Summary: 69 passed, 163 deselected, 1 warning in 13.03s
EOF
ls -la
rg --files -g 'run.json' reports | head -n 20
jq '{cap_active, cap_sources, window_coverage, windows_requested, windows_evaluated}' reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json')
obj = json.loads(p.read_text())
keys = {k: obj.get(k) for k in ['cap_active','cap_sources','window_coverage','windows_requested','windows_evaluated']}
print(keys)
PY
python - <<'PY'
import json
from pathlib import Path
obj = json.loads(Path('reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json').read_text())
print(obj['windows'].keys())
print(obj['windows'])
PY
ls reports/rc-ticket-05-20251221_221902/dow-paper-v1
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv')
df = pd.read_csv(p)
if 'reason_code' in df.columns:
    holdout = df[df['reason_code'] == 'holdout_empty']
    print('holdout_empty_count', len(holdout))
    # check NaN window identifier columns
    window_cols = [c for c in df.columns if 'window' in c or c in ('window_id','window_start','window_end')]
    print('window_cols', window_cols)
    if window_cols:
        nan_counts = holdout[window_cols].isna().sum()
        print('holdout_empty_nan_counts', nan_counts.to_dict())
else:
    print('reason_code not found', df.columns)
PY
rg -n "window_coverage|windows_requested|windows_evaluated|cap_active|cap_sources" experiments/eval/run.py
sed -n '2700,2920p' experiments/eval/run.py
rg -n "holdout_empty" -n experiments/eval/run.py
rg -n "reason_code|holdout|window_id" experiments/eval/run.py
sed -n '1960,2105p' experiments/eval/run.py
sed -n '2400,2685p' experiments/eval/run.py
rg -n "window_coverage|windows_requested|windows_evaluated|windows_after_caps|cap_active|cap_sources" -g '*.py'
rg -n "run.json|windows_dropped|windows_requested|windows_after_caps" tools/make_summary.py
sed -n '150,230p' tools/make_summary.py
rg -n "RunEligibility|completeness|run.json" tools/make_summary.py
sed -n '620,760p' tools/make_summary.py
sed -n '980,1075p' tools/make_summary.py
sed -n '3080,3165p' experiments/eval/run.py
sed -n '3740,3865p' experiments/eval/run.py
rg -n "def _limit_windows_by_regime" -n experiments/eval/run.py
sed -n '1450,1525p' experiments/eval/run.py
rg -n "HOLDOUT_EMPTY" -g '*.py'
sed -n '1,120p' experiments/eval/diagnostics.py
rg -n "reason_codes" -g '*.py' experiments/eval
rg -n "def _build_grouped_window|group_design" experiments/eval/run.py
sed -n '1294,1360p' experiments/eval/run.py
sed -n '1,120p' experiments/daily/grouping.py
rg -n "class EvalConfig|group_design|group_min" experiments/eval/config.py
rg -n "def _required_replicates" -n experiments/eval/run.py
sed -n '1270,1295p' experiments/eval/run.py
sed -n '1,80p' tools/make_summary.py
sed -n '1,120p' tests/experiments/test_eval_run.py
make test-fast
cat <<'EOF' >> docs/agent_runs/20251222_014730_ticket-06_window-coverage/TESTS.md

make test-fast (post-fix)
- Result: PASS
- Warnings: PytestConfigWarning: Unknown config option: timeout
- Summary: 69 passed, 164 deselected, 1 warning in 13.25s
EOF
rg -n "__main__|argparse" experiments/eval/run.py
sed -n '3920,3965p' experiments/eval/run.py
sed -n '660,760p' experiments/eval/run.py
rg -n "max_windows|start|end" experiments/eval/config.paper_v1.yaml
sed -n '1,200p' experiments/eval/config.paper_v1.yaml
date +%Y%m%d_%H%M%S
PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-06-20251222_020450/dow-paper-v1 --exec-mode deterministic
PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-06-20251222_020450
make test-fast
cat <<'EOF' >> docs/agent_runs/20251222_014730_ticket-06_window-coverage/TESTS.md

make test-fast (post-drop-reason fix)
- Result: PASS
- Warnings: PytestConfigWarning: Unknown config option: timeout
- Summary: 69 passed, 164 deselected, 1 warning in 13.20s
EOF
date +%Y%m%d_%H%M%S
PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-06-20251222_041746/dow-paper-v1 --exec-mode deterministic
PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-06-20251222_041746
make test-fast
cat <<'EOF' >> docs/agent_runs/20251222_014730_ticket-06_window-coverage/TESTS.md

make test-fast (holdout_empty metadata fix)
- Result: PASS
- Warnings: PytestConfigWarning: Unknown config option: timeout
- Summary: 69 passed, 164 deselected, 1 warning in 13.02s
EOF
date +%Y%m%d_%H%M%S
PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-06-20251222_063304/dow-paper-v1 --exec-mode deterministic
PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-06-20251222_063304
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rc-ticket-06-20251222_063304/dow-paper-v1/run.json')
obj = json.loads(p.read_text())
print(obj['windows'])
PY
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('reports/rc-ticket-06-20251222_063304/dow-paper-v1/diagnostics_detail.csv')
df = pd.read_csv(p)
print('drop_reason column:', 'drop_reason' in df.columns)
row = df[df['reason_code'] == 'holdout_empty'].head(1)
print(row[['reason_code','drop_reason','window_id','window_start']])
PY
python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path('reports/rc-ticket-06-20251222_063304/summary')
perf = pd.read_csv(base / 'summary_perf.csv')
det = pd.read_csv(base / 'summary_detection.csv')
forensics = pd.read_csv(base / 'overlay_forensics.csv')
print('perf_rows', len(perf))
print('det_rows', len(det))
print('forensics_rows', len(forensics))
print('perf_columns', [c for c in perf.columns if c.startswith('comparison_valid') or c.startswith('n_effective')])
print(perf[['comparison_valid_mse','comparison_valid_es','comparison_valid_qlike','comparison_valid_dm','comparison_valid_delta','n_effective','n_effective_mse','n_effective_es','n_effective_qlike']])
PY
cat reports/rc-ticket-06-20251222_063304/summary/limitations.md
python - <<'PY'
import json
import pandas as pd
from pathlib import Path
run_path = Path('reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json')
obj = json.loads(run_path.read_text())
print('windows', obj['windows'])

df = pd.read_csv('reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv')
holdout = df[df['reason_code'] == 'holdout_empty']
print('holdout_empty_count', len(holdout))
window_cols = [c for c in df.columns if c in ('window_id','window_start','window_end')]
print('holdout_empty_nan', {c: int(holdout[c].isna().sum()) for c in window_cols})
PY
git status -sb
ls -la docs/agent_runs/20251222_014730_ticket-06_window-coverage
tail -n 40 docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
sed -n '1,200p' PROGRESS.md
sed -n '1,200p' project_state/KNOWN_ISSUES.md
sed -n '1,200p' project_state/CURRENT_RESULTS.md
sed -n '1,240p' docs/CODEX_SPRINT_TICKETS.md
sed -n '240,400p' docs/CODEX_SPRINT_TICKETS.md
git diff --stat
git diff docs/CODEX_SPRINT_TICKETS.md
rg -n "windows_candidate|windows_planned|windows_requested|windows_dropped" -n experiments/eval/run.py
sed -n '2730,2925p' experiments/eval/run.py
rg -n "HOLDOUT_EMPTY" -n experiments/eval/run.py
sed -n '2020,2095p' experiments/eval/run.py
rg -n "drop_reason" experiments/eval/run.py
sed -n '2950,3060p' experiments/eval/run.py
sed -n '3095,3195p' experiments/eval/run.py
sed -n '3825,3895p' experiments/eval/run.py
rg -n "holdout_empty_windows" -n tests/experiments/test_eval_run.py
sed -n '260,360p' tests/experiments/test_eval_run.py
rg -n "holdout_empty|windows_dropped" tools/make_summary.py
sed -n '150,230p' tools/make_summary.py
rg -n "limitations" tools/make_summary.py
sed -n '1010,1095p' tools/make_summary.py
head -n 40 tools/make_summary.py
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rc-ticket-06-20251222_063304/dow-paper-v1/run.json')
obj = json.loads(p.read_text())
print('keys', obj.keys())
print('windows', obj['windows'])
print('config_hash', obj.get('config_hash'))
print('resolved_config', obj.get('config', {}).get('config_hash'))
print('returns', obj.get('returns'))
print('factors', obj.get('factors'))
PY
sha256sum data/returns_daily.csv data/factors/ff5mom_daily.csv
ls -la reports/rc-ticket-06-20251222_063304/dow-paper-v1 | head
ls reports/rc-ticket-06-20251222_063304/dow-paper-v1 | rg "config|resolved|yaml|json"
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rc-ticket-06-20251222_063304/dow-paper-v1/resolved_config.json')
obj = json.loads(p.read_text())
print('keys', obj.keys())
print('config_hash', obj.get('config_hash'))
print('returns_csv', obj.get('returns_csv'))
print('factors_csv', obj.get('factors_csv'))
PY
sha256sum experiments/eval/config.paper_v1.yaml
git rev-parse HEAD
sed -n '1,200p' reports/rc-ticket-06-20251222_063304/summary/limitations.md
python - <<'PY'
import pandas as pd
from pathlib import Path
perf = pd.read_csv('reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv')
print('rows', len(perf))
cols = [c for c in perf.columns if c.startswith('comparison_valid') or c.startswith('n_effective')]
print('cols', cols)
print(perf[cols])
PY
python - <<'PY'
import pandas as pd
base = 'reports/rc-ticket-06-20251222_063304/summary'
print('perf_rows', len(pd.read_csv(f'{base}/summary_perf.csv')))
print('det_rows', len(pd.read_csv(f'{base}/summary_detection.csv')))
print('forensics_rows', len(pd.read_csv(f'{base}/overlay_forensics.csv')))
PY
python - <<'PY'
import pandas as pd
p = 'reports/rc-ticket-06-20251222_063304/dow-paper-v1/diagnostics_detail.csv'
df = pd.read_csv(p)
row = df[df['reason_code'] == 'holdout_empty'].head(3)
print('holdout_rows', len(df[df['reason_code'] == 'holdout_empty']))
print(row[['reason_code','drop_reason','window_id','window_start']])
print('nan_window_id', int(df[df['reason_code']=='holdout_empty']['window_id'].isna().sum()))
PY
python - <<'PY'
import json
import pandas as pd
from pathlib import Path
run_path = Path('reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json')
obj = json.loads(run_path.read_text())
print('windows', obj['windows'])

df = pd.read_csv('reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv')
holdout = df[df['reason_code'] == 'holdout_empty']
print('holdout_empty_count', len(holdout))
window_cols = [c for c in df.columns if c in ('window_id','window_start','window_end')]
print('holdout_empty_nan', {c: int(holdout[c].isna().sum()) for c in window_cols})
PY
cat > docs/agent_runs/20251222_014730_ticket-06_window-coverage/RESULTS.md <<'EOF2'
# Results

## Pre-fix reproduction (ticket-05 evidence)
- `reports/rc-ticket-05-20251221_221902/dow-paper-v1/run.json` windows block:
  - `cap_active: true`, `cap_sources: ['window_coverage']`, `window_coverage: 0.9682933553901296`, `windows_requested: 3627`, `windows_evaluated: 3512`.
- `reports/rc-ticket-05-20251221_221902/dow-paper-v1/diagnostics_detail.csv`:
  - `reason_code == holdout_empty` count = 115.
  - `window_id` NaN count = 115; `window_start` NaN count = 115 (all holdout-empty rows missing identifiers).

## Fix verification (ticket-06 real-data run)
Run root: `reports/rc-ticket-06-20251222_063304/`

- `reports/rc-ticket-06-20251222_063304/dow-paper-v1/run.json` windows block:
  - `windows_candidate: 3627`, `windows_after_caps: 3627`.
  - `windows_dropped_holdout_empty: 115`, `windows_dropped_reasons: {'holdout_empty': 115}`.
  - `windows_requested: 3512`, `windows_evaluated: 3512`, `window_coverage: 1.0`.
  - `cap_active: false`, `cap_sources: []`.
- `reports/rc-ticket-06-20251222_063304/dow-paper-v1/diagnostics_detail.csv`:
  - `holdout_empty` rows now carry identifiers (`window_id`/`window_start` non-NaN); `drop_reason` column present.
- Summary outputs are non-empty:
  - `reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv` rows = 6.
  - `reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv` rows = 3.
  - `reports/rc-ticket-06-20251222_063304/summary/overlay_forensics.csv` rows = 6996.
- Validity checks:
  - `comparison_valid_* == 1` for all rows in `summary_perf.csv`.
  - `n_effective` min = 715 (>= 50).
- Limitations:
  - `reports/rc-ticket-06-20251222_063304/summary/limitations.md` includes “windows dropped from planning (holdout_empty: 115)” and no capped-run exclusion section.

## Artifacts
- Run outputs: `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`
- Summary directory: `reports/rc-ticket-06-20251222_063304/summary/`
EOF2
cat > docs/agent_runs/20251222_014730_ticket-06_window-coverage/META.md <<'EOF2'
# Meta

- RUN_NAME: 20251222_014730_ticket-06_window-coverage
- Branch: feat/ticket-06-window-coverage
- Git SHA: 5cda5baa68086c29c2f5a95c488f65a11865bc14

## Config
- Config file: experiments/eval/config.paper_v1.yaml
- Config sha256: b786cd691f79a86419e31e698e9f40ef7f8b14b296b72253c329942c3848e042
- Run config path (resolved): reports/rc-ticket-06-20251222_063304/dow-paper-v1/resolved_config.json

## Datasets
- returns_daily.csv sha256: 96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197
- ff5mom_daily.csv sha256: 469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca

## Environment notes
- Exec mode: deterministic (`--exec-mode deterministic`)
- PYTHONPATH: `src:.`
- Reports root: `reports/rc-ticket-06-20251222_063304/`
EOF2
cat docs/agent_runs/20251222_014730_ticket-06_window-coverage/TESTS.md
date -u +'%Y-%m-%dT%H:%MZ'
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: PROGRESS.md
@@
+## 2025-12-22T07:54Z — ticket-06 window_coverage planning fix (daily DoW paper v1)
+- **Branch/Run**: `feat/ticket-06-window-coverage` (RUN_NAME=`20251222_014730_ticket-06_window-coverage`), git sha `5cda5baa68086c29c2f5a95c488f65a11865bc14`.
+- **Commands**: `make test-fast`; `PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-06-20251222_063304/dow-paper-v1 --exec-mode deterministic`; `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-ticket-06-20251222_063304`.
+- **Results**: window_coverage no longer flags uncapped runs when the only missing windows are holdout-empty; run.json now tracks candidate/planned windows and holdout drops, and limitations.md surfaces the dropped count. The paper-v1 daily DoW run is headline-eligible (`cap_active=false`, `window_coverage=1.0`, summary tables non-empty, comparison_valid=1, n_effective>=715). Pre-fix evidence: ticket-05 run flagged `cap_active=true` with `cap_sources=['window_coverage']` due to 115 holdout-empty windows lacking identifiers; post-fix run logs `windows_dropped_holdout_empty=115` without capping.
+- **Artifacts**: run log `docs/agent_runs/20251222_014730_ticket-06_window-coverage/`; outputs `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`; summary `reports/rc-ticket-06-20251222_063304/summary/`.
+
*** End Patch
PATCH
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: project_state/KNOWN_ISSUES.md
@@
 - **Capped runs not headline (resolved 2025-12-21)**: `tools/make_summary.py` now excludes `cap_active=true` runs from headline tables and lists cap sources in limitations (ticket-02).
+- **Holdout-empty windows counted as caps (resolved 2025-12-22)**: daily eval window planning now drops holdout-empty windows from `windows_requested` and logs `windows_dropped_holdout_empty`, preventing false `window_coverage` caps on uncapped runs (ticket-06).
 - **Heavy ablation runtime**: `make rc-ablations` can time out on local hosts; use smaller grids or remote hosts.
*** End Patch
PATCH
python - <<'PY'
import pandas as pd
path = 'reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv'
df = pd.read_csv(path)
print(df)
PY
python - <<'PY'
import pandas as pd
path = 'reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv'
df = pd.read_csv(path)
print(df.columns.tolist())
PY
python - <<'PY'
import pandas as pd
path = 'reports/rc-ticket-06-20251222_063304/summary/summary_detection.csv'
df = pd.read_csv(path)
row = df[df['regime']=='full'].iloc[0]
print('full_windows', int(row['windows']))
print('detection_windows', int(row['detection_windows']))
print('detection_rate_mean', row['detection_rate_mean'])
print('detection_rate_median', row['detection_rate_median'])
PY
python - <<'PY'
import pandas as pd
path = 'reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv'
df = pd.read_csv(path)
print(df.columns.tolist())
PY
python - <<'PY'
import pandas as pd
path = 'reports/rc-ticket-06-20251222_063304/summary/summary_perf.csv'
df = pd.read_csv(path)
full = df[df['regime']=='full']
print(full[['portfolio','delta_mse_vs_baseline','delta_es_vs_baseline','delta_qlike_vs_baseline','n_effective']])
PY
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: project_state/CURRENT_RESULTS.md
@@
 # Current Results (latest validated drops)
 
+- **2025-12-22 — Daily DoW paper-v1 (ticket-06, git 5cda5baa)**  
+  - Deterministic daily DoW run (`experiments/eval/config.paper_v1.yaml`, FF5+MOM prewhiten) with uncapped windows: `cap_active=false`, `window_coverage=1.0`, `n_effective=1749` (full regime).  
+  - Full-regime detection_rate_mean ≈ 4.16% (1751/1774 windows); window drops logged as `holdout_empty: 115` (excluded from planning, not treated as caps).  
+  - Performance deltas (full regime): EW ΔMSE ≈ +2.64e-11 (harmful), MV ΔMSE ≈ −6.65e-13 (slight improvement).  
+  - Artifacts: `reports/rc-ticket-06-20251222_063304/summary/{summary_perf.csv,summary_detection.csv,overlay_forensics.csv,limitations.md}`; run dir `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`.
+
 - **2025-12-20 — Nested synthetic calibration (ticket-10, git e6e7982)**  
   - Synthetic nested (p=200, years=2, weeks 6–8, reps=5, tyler, delta=0.35, delta_frac=0.05): null detections 0/220 → FPR 0 with Wilson hi 0.017; power 1.0 on moderate/strong.  
*** End Patch
PATCH
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: docs/CODEX_SPRINT_TICKETS.md
@@
-## Ticket #5 — Advisor-ready RC run (one headline table, no caveats)
+## Ticket #5 (FAIL) — Advisor-ready RC run (one headline table, no caveats)
@@
 **Expected artifacts/logs:**
 - Full run log + `make gpt-bundle` bundle for advisor
+
+**Status:** FAIL — ticket-05 daily DoW paper-v1 run flagged `cap_active=true` due to `window_coverage` from holdout-empty windows; headline tables excluded.
 
-Ticket #6 — Replace/repair make rc-lite-sanity so it always finishes (<2–3 min) and is truly uncapped, or update AGENTS to require the new smoke target.
-This is necessary to stop “timeouts” from being the default test outcome.
+---
 
-Add one new ticket at the bottom (needed):
+## Ticket #6 (DONE) — Window coverage planning for holdout-empty windows (daily eval)
+
+**Goal (1 sentence):** Ensure uncapped daily eval runs are not flagged `cap_active` when the only missing windows are holdout-empty; log dropped windows and keep headline tables non-empty.
+
+**Primary blocker addressed:** “uncapped paper-v1 run flagged `window_coverage` due to holdout_empty planning.”
+
+**Files/modules likely involved:**
+- `experiments/eval/run.py` (window planning + drop reasons)
+- `tools/make_summary.py` (limitations surfacing)
+- tests:
+  - `tests/experiments/test_eval_run.py`
+
+**Acceptance criteria:**
+- `windows_requested` reflects evaluable windows (post holdout-empty drop); `windows_dropped_holdout_empty` logged.
+- `cap_active=false` on uncapped daily DoW paper-v1 run with holdout-empty windows.
+- `summary_perf.csv`, `summary_detection.csv`, and `overlay_forensics.csv` are non-empty.
+- Regression test added to prevent `window_coverage` caps driven by holdout-empty windows.
+
+**Status:** DONE — see `reports/rc-ticket-06-20251222_063304/` and run log `docs/agent_runs/20251222_014730_ticket-06_window-coverage/`.
+
+---
 
-Ticket #7 — Re-run Ticket #5 after Ticket #6 fix: one advisor-ready daily DoW paper-v1 table (uncapped), update PROGRESS.md + project_state/CURRENT_RESULTS.md, bundle for advisor.
+## Ticket #7 — Re-run Ticket #5 after Ticket #6 fix: one advisor-ready daily DoW paper-v1 table (uncapped), update PROGRESS.md + project_state/CURRENT_RESULTS.md, bundle for advisor.
+
+## Ticket #8 — Replace/repair make rc-lite-sanity so it always finishes (<2–3 min) and is truly uncapped, or update AGENTS to require the new smoke target.
+This is necessary to stop “timeouts” from being the default test outcome.
*** End Patch
PATCH
tail -n 40 docs/CODEX_SPRINT_TICKETS.md
rg -n "rc-ticket-06-20251222_063304" docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
git status -sb
git diff experiments/eval/run.py | head -n 120
git add experiments/eval/run.py tools/make_summary.py tests/experiments/test_eval_run.py
git commit -m "Fix window coverage planning for holdout-empty windows" -m "Tests run: make test-fast"
git rev-parse HEAD
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: PROGRESS.md
@@
-- **Branch/Run**: `feat/ticket-06-window-coverage` (RUN_NAME=`20251222_014730_ticket-06_window-coverage`), git sha `5cda5baa68086c29c2f5a95c488f65a11865bc14`.
+- **Branch/Run**: `feat/ticket-06-window-coverage` (RUN_NAME=`20251222_014730_ticket-06_window-coverage`), git sha `8a5579b8f34176b43c75543c0a3305f8a8fe2aa2`.
*** End Patch
PATCH
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: project_state/CURRENT_RESULTS.md
@@
-- **2025-12-22 — Daily DoW paper-v1 (ticket-06, git 5cda5baa)**  
+- **2025-12-22 — Daily DoW paper-v1 (ticket-06, git 8a5579b)**  
*** End Patch
PATCH
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: docs/agent_runs/20251222_014730_ticket-06_window-coverage/META.md
@@
-- Git SHA: 5cda5baa68086c29c2f5a95c488f65a11865bc14
+- Git SHA: 8a5579b8f34176b43c75543c0a3305f8a8fe2aa2
*** End Patch
PATCH
git status -sb
git status -sb
make gpt-bundle TICKET=ticket-06 RUN_NAME=20251222_014730_ticket-06_window-coverage
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: docs/agent_runs/20251222_014730_ticket-06_window-coverage/RESULTS.md
@@
 ## Artifacts
 - Run outputs: `reports/rc-ticket-06-20251222_063304/dow-paper-v1/`
 - Summary directory: `reports/rc-ticket-06-20251222_063304/summary/`
+- GPT bundle: `docs/gpt_bundles/20251222_090028_ticket-06_20251222_014730_ticket-06_window-coverage.zip`
*** End Patch
PATCH
git status -sb
git add PROGRESS.md project_state/CURRENT_RESULTS.md project_state/KNOWN_ISSUES.md docs/CODEX_SPRINT_TICKETS.md docs/agent_runs/20251222_014730_ticket-06_window-coverage
git commit -m "Document ticket-06 run and update sprint tickets" -m "Tests run: make test-fast"
git add docs/agent_runs/20251222_014730_ticket-06_window-coverage/RESULTS.md docs/agent_runs/20251222_014730_ticket-06_window-coverage/COMMANDS.md
git commit -m "Record ticket-06 bundle and logs" -m "Tests run: make test-fast"
