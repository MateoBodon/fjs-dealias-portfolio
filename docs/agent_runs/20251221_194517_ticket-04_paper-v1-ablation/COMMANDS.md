# Commands
- git status --porcelain
- git checkout -b feat/ticket-04-paper-v1-ablation
- date +%Y%m%d_%H%M%S
- rg --files -g 'experiments/eval/*.yaml'
- sed -n '1,220p' experiments/eval/config.yaml
- rg -n "config" experiments/eval -g '*.py'
- sed -n '1,240p' experiments/eval/config.py
- rg -n "paper|v1|min_comparison_windows|comparison" docs/PLAN_OF_RECORD.md
- sed -n '160,240p' docs/PLAN_OF_RECORD.md
- sed -n '1,260p' tools/make_summary.py
- rg -n "summary_perf|summary_detection|summary/summary" -n tools/make_summary.py
- sed -n '840,1030p' tools/make_summary.py
- rg -n "delta_mse_vs_baseline|summary_perf" -n tools/make_summary.py
- sed -n '520,840p' tools/make_summary.py
- rg -n "overlay" experiments/eval/run.py
- rg -n "overlay.*off|off\\s*overlay|overlay off|--overlay" -n experiments/eval/run.py
- sed -n '1680,1790p' experiments/eval/run.py
- sed -n '1120,1185p' experiments/eval/run.py
- rg -n "overlay_allowed|overlay_delta" experiments/eval/run.py
- rg -n "def apply_overlay|class OverlayConfig|delta" -n src/fjs/overlay.py
- sed -n '1,200p' src/fjs/overlay.py
- sed -n '400,520p' src/fjs/overlay.py
- rg -n "def dealias_search|delta" -n src/fjs/dealias.py
- sed -n '520,620p' src/fjs/dealias.py
- rg -n "gate_mode" -n src/fjs/gating.py
- rg --files -g 'src/fjs/*.py'
- sed -n '1,240p' src/fjs/gating.py
- rg -n "detection_rate|detections" experiments/eval/run.py
- sed -n '2080,2235p' experiments/eval/run.py
- ls -la data | head -n 200
- ls -la data/meta
- rg --files -g '*returns*' data
- rg --files -g '*.csv'
- rg -n "returns.*tiny|tiny.*returns" -S
- ls -la tests/data
- python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('reports/rc-20251103/sample_returns.csv')
print(path.exists())
if path.exists():
    df = pd.read_csv(path, nrows=5)
    print(df.head())
    print(df.columns[:5])
    print('rows', sum(1 for _ in open(path)) - 1)
PY
- ls -la
- rg --files -g '*completeness*'
- sed -n '1,220p' src/meta/completeness.py
- rg -n "cap_active|max_windows|start|end" experiments/eval/run.py
- sed -n '2720,2905p' experiments/eval/run.py
- rg --files -g '*returns*csv' reports | head -n 50
- python - <<'PY'
import pandas as pd
from pathlib import Path
path = Path('data/prices_sample.csv')
print(path.exists())
if path.exists():
    df = pd.read_csv(path, nrows=5)
    print(df.head())
    import csv
    with open(path, 'r', encoding='utf-8') as f:
        rows = sum(1 for _ in f) - 1
    print('rows', rows, 'cols', len(df.columns))
PY
- rg -n "overlay|ablate|rc-" Makefile
- sed -n '120,220p' Makefile
- sed -n '300,380p' Makefile
- rg -n "RC_RETURNS|RC_FACTORS|RC_" Makefile | head -n 80
- sed -n '650,940p' experiments/eval/run.py
- ls -la tests
- ls -la tests/tools
- sed -n '1,220p' tests/tools/test_make_summary.py
- sed -n '1,200p' tests/tools/test_summarize_rc_sanity.py
- sed -n '1,200p' docs/CODEX_SPRINT_TICKETS.md
- sed -n '1,200p' PROGRESS.md
- rg -n "ablation|paper_v1" -n tools
- rg -n "summary_perf|summary_detection" tools/build_memo.py
- sed -n '840,930p' tools/build_memo.py
- rg -n "summary_perf\\.csv|summary_detection\\.csv" -S
- sed -n '220,360p' experiments/eval/run.py
- apply_patch (tools/make_summary.py)
- cat > experiments/eval/config.paper_v1.yaml <<'EOF' ... EOF
- cat > tools/paper_v1_ablation.py <<'EOF' ... EOF
- cat > tests/tools/test_paper_v1_ablation.py <<'EOF' ... EOF
- apply_patch (Makefile)
- python - <<'PY'
import pandas as pd
from pathlib import Path

source = Path('data/returns_daily.csv')
if not source.exists():
    raise SystemExit('missing data/returns_daily.csv')

out_path = Path('reports/fixtures/returns_daily_small.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

# Load minimal columns
cols = ["date", "ticker", "ret"]
df = pd.read_csv(source, usecols=cols, parse_dates=["date"])

# Use the most recent 220 trading dates to maximize coverage
unique_dates = sorted(df["date"].dropna().unique())
if len(unique_dates) < 220:
    raise SystemExit(f"Not enough dates ({len(unique_dates)}) for small dataset")
selected_dates = unique_dates[-220:]
subset = df[df["date"].isin(selected_dates)].copy()

# Prefer tickers with full coverage across selected dates
date_count = len(selected_dates)
counts = subset.groupby("ticker")["date"].nunique().sort_values(ascending=False)
full_tickers = counts[counts == date_count].index.tolist()
if full_tickers:
    selected_tickers = full_tickers[:60]
else:
    selected_tickers = counts.head(60).index.tolist()

small = subset[subset["ticker"].isin(selected_tickers)].copy()
small.to_csv(out_path, index=False)

print(f"Wrote {out_path} rows={len(small)} tickers={small['ticker'].nunique()} dates={small['date'].nunique()}")
print(f"Date range: {small['date'].min().date()} to {small['date'].max().date()}")
PY
- make test-fast
- apply_patch (Makefile: scm shrinker -> sample)
- apply_patch (experiments/eval/run.py: short-circuit detect_spikes when q_max<=0)
- apply_patch (experiments/eval/config.paper_v1.yaml: overlay_a_grid 120 -> 60)
- python - <<'PY'
import pandas as pd
from pathlib import Path

source = Path('data/returns_daily.csv')
if not source.exists():
    raise SystemExit('missing data/returns_daily.csv')

out_path = Path('reports/fixtures/returns_daily_small.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

cols = ["date", "ticker", "ret"]
df = pd.read_csv(source, usecols=cols, parse_dates=["date"])

unique_dates = sorted(df["date"].dropna().unique())
if len(unique_dates) < 196:
    raise SystemExit(f"Not enough dates ({len(unique_dates)}) for small dataset")
selected_dates = unique_dates[-196:]
subset = df[df["date"].isin(selected_dates)].copy()

# Prefer tickers with full coverage across selected dates
count_required = len(selected_dates)
counts = subset.groupby("ticker")["date"].nunique().sort_values(ascending=False)
full_tickers = counts[counts == count_required].index.tolist()
if full_tickers:
    selected_tickers = full_tickers[:60]
else:
    selected_tickers = counts.head(60).index.tolist()

small = subset[subset["ticker"].isin(selected_tickers)].copy()
small.to_csv(out_path, index=False)

print(f"Wrote {out_path} rows={len(small)} tickers={small['ticker'].nunique()} dates={small['date'].nunique()}")
print(f"Date range: {small['date'].min().date()} to {small['date'].max().date()}")
PY
- apply_patch (experiments/eval/config.paper_v1.yaml: overlay_a_grid 60 -> 30)
- EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate (failed: invalid shrinker)
- EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate (interrupted)
- EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate (interrupted)
- EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate (interrupted)
- EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate (interrupted)
- EXEC_MODE=deterministic PAPER_V1_RETURNS=reports/fixtures/returns_daily_small.csv make rc-paper-v1-ablate (completed; reports/rc-paper-v1-ablate-20251221_205751)
- make test-fast (rerun after run.py/config changes)
- git rev-parse HEAD
- sha256sum experiments/eval/config.paper_v1.yaml reports/fixtures/returns_daily_small.csv | cat
- apply_patch (docs/CODEX_SPRINT_TICKETS.md: mark Ticket #1/#4 DONE)
- date -u +%Y-%m-%dT%H:%MZ
- apply_patch (PROGRESS.md: add ticket-04 entry)
- git diff > docs/agent_runs/20251221_194517_ticket-04_paper-v1-ablation/DIFF.patch
- ls -la reports/rc-paper-v1-ablate-20251221_201330
- ls -la reports/rc-paper-v1-ablate-20251221_201330/scm_off | head -n 50
- ls -la reports/rc-paper-v1-ablate-20251221_201330 | sed -n '1,20p'
- ls -la reports/rc-paper-v1-ablate-20251221_202528
- ls -la reports/rc-paper-v1-ablate-20251221_202528/scm_off | head -n 20
- ls -la reports/rc-paper-v1-ablate-20251221_202528/scm_on | grep run
- ls -la reports/rc-paper-v1-ablate-20251221_203014
- ls -la reports/rc-paper-v1-ablate-20251221_203014/scm_on | grep run
- ls -lh data/factors/ff5mom_daily.csv
- wc -l data/returns_daily.csv
- head -n 2 data/returns_daily.csv
- rg -n "returns_csv" -n experiments/eval/run.py
- rg -n "load_daily_panel" -S
- rg -n "DailyLoaderConfig|load_daily_panel" -S src
- rg -n "DailyLoaderConfig|load_daily_panel" -n src/data/panels.py
- ls -la data/loader.py
- ls -la src/data
- sed -n '1,200p' src/data/__init__.py
- sed -n '1520,1585p' experiments/eval/run.py
- python - <<'PY'
import json
from pathlib import Path

root = Path('reports/rc-paper-v1-ablate-20251221_205751')
subdirs = ['scm_off','scm_on','oas_off','oas_on','rie_off','rie_on']
for name in subdirs:
    run_json = root / name / 'run.json'
    if not run_json.exists():
        print(name, 'MISSING run.json')
        continue
    payload = json.loads(run_json.read_text())
    windows = payload.get('windows', {})
    cap_active = windows.get('cap_active')
    cap_sources = windows.get('cap_sources')
    print(name, 'cap_active=', cap_active, 'cap_sources=', cap_sources)
PY
- python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('reports/rc-paper-v1-ablate-20251221_205751')
perf = root / 'summary' / 'summary_perf.csv'
det = root / 'summary' / 'summary_detection.csv'
abl = root / 'summary' / 'paper_v1_ablation.csv'
for path in [perf, det, abl]:
    if not path.exists():
        print(path, 'MISSING')
        continue
    df = pd.read_csv(path)
    print(path.name, 'rows', len(df))
PY
- sed -n '1,200p' reports/rc-paper-v1-ablate-20251221_205751/summary/limitations.md
- head -n 10 reports/rc-paper-v1-ablate-20251221_205751/summary/paper_v1_ablation.csv
- git status --porcelain
- git add Makefile experiments/eval/config.paper_v1.yaml experiments/eval/run.py tools/make_summary.py tools/paper_v1_ablation.py tests/tools/test_paper_v1_ablation.py
- git commit -m "feat: add paper v1 ablation runner" -m "Tests run: make test-fast"
