# Commands

- ls
- rg --files -g 'PROJECT.md' -g 'AGENTS.md' -g 'PROGRESS.md'
- cat AGENTS.md
- cat PROGRESS.md
- git status -sb
- git diff --stat
- python3 /root/.codex/skills/repo-bootstrap/scripts/bootstrap_repo.py
- cat docs/tickets/FJS-TKT-022.md
- make test-fast
- make run:equity_nested_smoke_tiny
- ls -lt experiments/equity_panel/outputs_nested_smoke_tiny
- ls -lt experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff
- python - <<'PY'
import pandas as pd
from pathlib import Path
from experiments.equity_panel.reasons import SkipReasonPrimary
out_dir = Path('experiments/equity_panel/outputs_nested_smoke_tiny/nested_J5_solver-auto_est-dealias_prep-prewhiten_modeoff')
for name in ['gating_diagnostics.csv','detection_summary.csv']:
    path = out_dir / name
    df = pd.read_csv(path)
    print(name, 'rows=', len(df))
    if 'skip_reason_primary' in df.columns:
        print('  diagnostic_failure rows:', (df['skip_reason_primary'] == str(SkipReasonPrimary.DIAGNOSTIC_FAILURE)).sum())
PY
- python3 tools/agentic/project_state_refresh.py --zip
- ls docs/DOCS_AND_LOGGING_SYSTEM.md docs/CODEX_SPRINT_TICKETS.md project_state/CURRENT_RESULTS.md project_state/KNOWN_ISSUES.md project_state/CONFIG_REFERENCE.md
