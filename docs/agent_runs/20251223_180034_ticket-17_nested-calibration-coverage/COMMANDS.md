git status -sb
git checkout -b codex/ticket-17-nested-calibration-coverage
date +%Y%m%d_%H%M%S
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage
mkdir -p docs/agent_runs/$RUN_NAME
: > docs/agent_runs/$RUN_NAME/COMMANDS.md
: > docs/agent_runs/$RUN_NAME/RESULTS.md
: > docs/agent_runs/$RUN_NAME/TESTS.md
: > docs/agent_runs/$RUN_NAME/META.md
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage
cat > docs/agent_runs/$RUN_NAME/PROMPT.md <<'EOF'
Implement **Ticket #17 — Nested calibration grid coverage** in repo `fjs-dealias-portfolio`.

CRITICAL: Follow `AGENTS.md` (stop-the-line rules) and `docs/DOCS_AND_LOGGING_SYSTEM.md` (logging contract). Do not create “fake fixes” (e.g., mapping missing (p,T) to a random existing cell without validating null-FPR).

Do NOT write a long upfront plan. Start by inspecting current behavior, then implement, test, and document end-to-end.

### Branch + run log (required)
1) Create a feature branch:
   - `git checkout -b codex/ticket-17-nested-calibration-coverage`

2) Set:
   - `RUN_NAME=$(date +%Y%m%d_%H%M%S)_ticket-17_nested-calibration-coverage`
   - Create `docs/agent_runs/$RUN_NAME/` and populate (REQUIRED):
     - `PROMPT.md` (paste this prompt verbatim)
     - `COMMANDS.md` (every command you run, in order)
     - `RESULTS.md` (what changed + artifact paths + any failures)
     - `TESTS.md` (exact tests run + pass/fail)
     - `META.md` (git SHA before/after, branch, dirty-at-start, dataset ids/hashes used, config hashes)

3) Commit in small logical chunks. Every commit body MUST include:
   - `Tests: <exact commands>`

### Acceptance criteria (must satisfy all)
From `docs/CODEX_SPRINT_TICKETS.md` Ticket #17:
- `make run:equity_nested_smoke_tiny` produces windows that do NOT skip with `calibration_missing_p_T`.
- `calibration/nested_edge_delta_thresholds.json` includes audit metadata (run_name, timestamp, git_sha, config_hash) and thresholds for newly required grid cells.
- Synthetic nested null-FPR at the operating point remains ≤ target (2% is fine) for newly added (p,T) cells.

### Work steps (do these, in this order)
A) Reproduce current failure (baseline evidence)
1) Run and record:
   - `make test-fast` (only if already quick; otherwise do later after code edits)
   - `EXEC_MODE=deterministic make run:equity_nested_smoke_tiny`
2) Locate outputs under `experiments/equity_panel/outputs_nested_smoke_tiny/...`
3) Extract the observed (p,T) that cause `calibration_missing_p_T` (expect p≈188, T∈{70,80}).
4) In `docs/agent_runs/$RUN_NAME/RESULTS.md`, record:
   - the pre-fix skip_reason histogram / summary showing `calibration_missing_p_T`
   - the observed (p,T) pairs

B) Understand current calibration schema + lookup strictness
1) Inspect:
   - `calibration/nested_edge_delta_thresholds.json` (existing grid + metadata)
   - `src/fjs/gating.py` (how nested looks up thresholds; how p/T are computed/bucketed)
   - `experiments/synthetic/nested_killtest.py` and `experiments/synthetic/config.nested.killtest.yaml` (how thresholds are generated)
2) Determine precisely why (p≈188, T∈{70,80}) is missing:
   - Is p binned? Are you rounding? Is T “effective T” vs “window weeks” mismatch?
   - Fix the *root cause*, not the symptom.

C) Extend calibration coverage (NO fake fixes)
Preferred approach: **actually calibrate** the missing cells.
1) Update the nested killtest config and/or generator so it can generate thresholds for the missing (p,T) cells.
   - Add cells for p around 188 (exact 188 if supported; otherwise an explicit, documented binning rule that guarantees the real-data p maps to a calibrated cell).
   - Add T={70,80} (or the exact T definition used in gating).
2) Run:
   - `python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/$RUN_NAME --calibration-out calibration/nested_edge_delta_thresholds.json`
3) Ensure the calibration JSON is updated in-place and includes audit metadata:
   - run_name, timestamp, git sha, config hash (sha256 of the written calibration file, or whatever the repo standard is)
4) Validate null-FPR for the new (p,T) cells at the operating point:
   - If the harness is configurable, choose trials high enough that a ≤2% target is meaningful (document exact trials used).
   - Record null-FPR table(s) in `reports/synthetic/nested_killtest/$RUN_NAME/` and summarize in RESULTS.md.

If you implement any fallback/approximation (e.g., nearest-neighbor):
- It MUST be explicit (logged reason code) and MUST come with synthetic evidence that null-FPR is still controlled.
- Do not silently map.

D) Update tests (fast + deterministic)
Add/extend tests to prevent regression:
- `tests/synthetic/test_calibration.py`:
  - asserts calibration file has coverage for the observed (p,T) cells OR that lookup returns a calibrated operating point without “missing”.
- `tests/test_threshold_eval.py`:
  - ensure nested lookup does not return `calibration_missing_p_T` for those (p,T) pairs.

Run (REQUIRED):
- `make test-fast`

E) Re-run real-data tiny smoke (post-fix)
- `EXEC_MODE=deterministic make run:equity_nested_smoke_tiny`
Acceptance check:
- The outputs must show **no** `calibration_missing_p_T` skips.
- If windows still skip, the reason must be something else and must be explicitly attributed.

F) Update docs + provenance (required)
1) `PROGRESS.md`:
   - Add a Ticket-17 entry with branch, SHAs, exact commands, tests, and artifact paths:
     - `reports/synthetic/nested_killtest/$RUN_NAME/`
     - `experiments/equity_panel/outputs_nested_smoke_tiny/...`
2) `project_state/KNOWN_ISSUES.md`:
   - Remove or revise the “nested calibration grid lacks p≈188 coverage” issue once fixed.
3) If nested smoke is now unblocked and yields meaningful new validity info:
   - Update `project_state/CURRENT_RESULTS.md` with a concise entry (date + what changed).

G) Finish cleanly + bundle for review
1) Ensure `git status` is clean.
2) Save a patch:
   - `git diff > docs/agent_runs/$RUN_NAME/DIFF.patch`
3) Generate a GPT bundle:
   - `make gpt-bundle TICKET=ticket-17 RUN_NAME=$RUN_NAME`
4) In `docs/agent_runs/$RUN_NAME/RESULTS.md`, record the bundle path and (recommended) `unzip -l` listing.

DONE means:
- acceptance criteria satisfied,
- tests green,
- run log complete,
- docs updated,
- no silent fallback introduced.
EOF
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage
cat >> docs/agent_runs/$RUN_NAME/COMMANDS.md <<'EOF'
git status -sb
git checkout -b codex/ticket-17-nested-calibration-coverage
date +%Y%m%d_%H%M%S
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage
mkdir -p docs/agent_runs/$RUN_NAME
: > docs/agent_runs/$RUN_NAME/COMMANDS.md
: > docs/agent_runs/$RUN_NAME/RESULTS.md
: > docs/agent_runs/$RUN_NAME/TESTS.md
: > docs/agent_runs/$RUN_NAME/META.md
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage
cat > docs/agent_runs/$RUN_NAME/PROMPT.md <<'EOF'
...prompt...
EOF
EOF
sed -n '1,120p' docs/agent_runs/$RUN_NAME/COMMANDS.md
sed -n '1,200p' AGENTS.md
sed -n '1,240p' docs/DOCS_AND_LOGGING_SYSTEM.md
make test-fast
python - <<'PY'
import hashlib
from pathlib import Path
for path_str in [
    'experiments/synthetic/config.nested.killtest.yaml',
    'experiments/equity_panel/config.nested.smoke.tiny.yaml',
]:
    path = Path(path_str)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path_str, digest)
PY
python - <<'PY'
import json
from pathlib import Path
registry = json.loads(Path('data/registry.json').read_text())
print(registry['datasets']['data/returns_daily.csv']['sha256'])
PY
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1
EXEC_MODE=deterministic make run:equity_nested_smoke_tiny
ls -1t experiments/equity_panel/outputs_nested_smoke_tiny | head
ls -1t experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff | head
rg -n "calibration_missing_p_T" experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff
python - <<'PY'
import json
from pathlib import Path
path = Path('experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/summary.json')
with path.open() as f:
    data=json.load(f)
print('skip_reason_mode:', data.get('skip_reason_mode'))
print('skip_reason_hist:', data.get('skip_reason_histogram'))
PY
python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/gating_diagnostics.csv')
df = pd.read_csv(path)
print(df['skip_reason'].value_counts(dropna=False).to_string())
PY
python - <<'PY'
import json
from pathlib import Path
path = Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
print(data.keys())
print('meta', data.get('meta', {}).keys())
# print grid info
thresholds = data.get('thresholds', {})
print('edges', list(thresholds.keys())[:5])
edge='tyler'
if edge in thresholds:
    grid = thresholds[edge]
    print('tyler keys', list(grid.keys())[:5])
    # list p values
    ps = sorted(int(p) for p in grid.keys())
    print('p range', ps[:10], '...', ps[-5:])
    # for p near 188
    for p in [180, 188, 190, 200]:
        if str(p) in grid:
            Ts = sorted(int(t) for t in grid[str(p)].keys())
            print('p', p, 'Ts', Ts[:10], '...', Ts[-5:])
PY
python - <<'PY'
import json
from pathlib import Path
path = Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
thresholds = data.get('thresholds', {})
edge='tyler'
print('edges:', thresholds.keys())
if edge in thresholds:
    grid = thresholds[edge]
    # keys are like '200x60'
    print('num cells', len(grid))
    keys = sorted(grid.keys())
    print('first keys', keys[:10])
PY
rg -n "nested|edge_delta|calibration_missing_p_T" src/fjs/gating.py
sed -n '150,280p' src/fjs/gating.py
rg -n "lookup_calibrated_delta|calibration_missing_p_T|delta_frac" src/fjs/gating.py
rg -n "lookup_calibrated_delta|calibration_missing_p_T" src
sed -n '210,320p' src/fjs/overlay.py
rg -n "lookup_calibrated_delta|gate_delta_calibration" -n src/fjs/overlay.py
sed -n '320,420p' src/fjs/overlay.py
rg -n "groups|nested" experiments/equity_panel/run.py
sed -n '1480,1625p' experiments/equity_panel/run.py
rg -n "detect_spikes" experiments/equity_panel/run.py
rg -n "overlay|gating|detections" experiments/equity_panel/run.py
sed -n '1680,1845p' experiments/equity_panel/run.py
rg -n "n_fit_samples" -n experiments/equity_panel/run.py
python - <<'PY'
import json
from pathlib import Path
path = Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
for key in ['run_name','timestamp_utc','generated_at','git_sha','config_hash','metadata','schema_version','design','alpha','target_fpr']:
    if key in data:
        print(key, data[key])
print('keys', list(data.keys()))
PY
python - <<'PY'
import json
from pathlib import Path
path = Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
print('design_thresholds keys', data.get('design_thresholds', {}).keys())
print('design_thresholds nested type', type(data.get('design_thresholds', {}).get('nested')))
print('design_thresholds nested keys', list(data.get('design_thresholds', {}).get('nested', {}).keys())[:5])
PY
python - <<'PY'
import json
from pathlib import Path
path = Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
print('top-level thresholds keys', data.get('thresholds', {}).keys())
PY
python - <<'PY'
import json
from pathlib import Path
path=Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
print(data['thresholds']['tyler'].keys())
print(data['thresholds']['tyler']['200x70'])
PY
rg -n "achieved_fpr|design_thresholds|timestamp_utc|schema_version" -S
sed -n '1,140p' README.md
rg -n "config_hash" -S
sed -n '1,220p' docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/COMMANDS.md
sed -n '1,200p' tests/synthetic/test_calibration.py
sed -n '1,260p' tests/test_threshold_eval.py
sed -n '70,130p' experiments/synthetic/nested_killtest.py
sed -n '1,30p' experiments/synthetic/nested_killtest.py
sed -n '30,90p' experiments/synthetic/nested_killtest.py
rg -n "n_assets =" experiments/synthetic/nested_killtest.py
sed -n '290,420p' experiments/synthetic/nested_killtest.py
sed -n '300,460p' experiments/synthetic/nested_killtest.py
sed -n '460,620p' experiments/synthetic/nested_killtest.py
sed -n '560,760p' experiments/synthetic/nested_killtest.py
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage --calibration-out calibration/nested_edge_delta_thresholds.json
python - <<'PY'
import json
from pathlib import Path
path = Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
print('run_name', data.get('run_name'))
print('timestamp_utc', data.get('timestamp_utc'))
print('git_sha', data.get('git_sha'))
print('metadata config_hash', data.get('metadata', {}).get('config_hash'))
print('schema_version', data.get('schema_version'))
print('design_thresholds keys', data.get('design_thresholds', {}).keys())
print('tyler keys', sorted(list(data['thresholds']['tyler'].keys()))[:6])
print('has 188x70', '188x70' in data['thresholds']['tyler'])
print('has 188x80', '188x80' in data['thresholds']['tyler'])
PY
rg -n "run_name" -n experiments/synthetic/nested_killtest.py
sed -n '610,630p' experiments/synthetic/nested_killtest.py
RUN_NAME=20251223_180034_ticket-17_nested-calibration-coverage python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage --calibration-out calibration/nested_edge_delta_thresholds.json
python - <<'PY'
import json
from pathlib import Path
path=Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
print('run_name', data.get('run_name'))
print('metadata.run_name', data.get('metadata', {}).get('run_name'))
print('timestamp_utc', data.get('timestamp_utc'))
print('metadata.config_hash', data.get('metadata', {}).get('config_hash'))
PY
python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/summary.csv')
df = pd.read_csv(path)
print(df)
print('\\nnull rows p=188')
print(df[(df['scenario']=='null') & (df['p_assets']==188)].to_string(index=False))
PY
sed -n '1,20p' reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/summary.csv
python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/summary.csv')
df = pd.read_csv(path, keep_default_na=False)
null_df = df[(df['scenario']=='null') & (df['p_assets'].isin([188,200]))]
print(null_df[['edge_mode','scenario','p_assets','trials','detections','detection_rate','ci_low','ci_high']].to_string(index=False))
PY
python - <<'PY'
import json
from pathlib import Path
path=Path('calibration/nested_edge_delta_thresholds.json')
with path.open() as f:
    data=json.load(f)
entry = data['thresholds']['tyler']['188x70']
print(entry)
PY
EXEC_MODE=deterministic make run:equity_nested_smoke_tiny
rg -n "calibration_missing_p_T" experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff
python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/gating_diagnostics.csv')
df = pd.read_csv(path)
print(df['skip_reason'].value_counts(dropna=False).to_string())
PY
ls -1 reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage
python - <<'PY'
import json
from pathlib import Path
path=Path('reports/synthetic/nested_killtest/20251223_180034_ticket-17_nested-calibration-coverage/run.json')
with path.open() as f:
    data=json.load(f)
print('config_hash', data.get('config_hash'))
print('resolved_config_hash', data.get('resolved_config_hash'))
print('config_path', data.get('config_path'))
PY
sed -n '1,200p' project_state/KNOWN_ISSUES.md
sed -n '1,200p' project_state/CURRENT_RESULTS.md
sed -n '1,200p' PROGRESS.md
python - <<'PY'
import json
from pathlib import Path
path = Path('experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff/run_meta.json')
with path.open() as f:
    data=json.load(f)
keys = {k: data.get(k) for k in ['git_sha','git_dirty','data_paths','data_hashes','config_path','config_hash','cap_active','cap_sources']}
print(keys)
PY
sed -n '1,200p' experiments/equity_panel/config.nested.smoke.tiny.yaml
sed -n '1,200p' data/registry.json
make test-fast
