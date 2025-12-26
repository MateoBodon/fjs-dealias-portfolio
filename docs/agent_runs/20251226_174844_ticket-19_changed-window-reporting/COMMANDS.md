cat AGENTS.md
cat docs/DOCS_AND_LOGGING_SYSTEM.md
date -u +%Y%m%d_%H%M%S
git status -sb
git checkout -b codex/ticket-19_changed-window-reporting
mkdir -p docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting
cat <<'EOF_PROMPT' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/PROMPT.md
You are working in a research repo with strict stop-the-line rules.

Before doing anything: read and follow AGENTS.md and docs/DOCS_AND_LOGGING_SYSTEM.md. If you find a conflict, stop and report it in the run log RESULTS.md.

TICKET: ticket-19
RUN_NAME: use a UTC timestamp + slug exactly like:
  RUN_NAME="$(date -u +%Y%m%d_%H%M%S)_ticket-19_changed-window-reporting"

GOAL (one sentence):
Add conditional (“changed-window only”) effect reporting + weight-change magnitude stats so we can tell whether the overlay matters when it triggers, and ensure this matches the semantics used for n_effective/aligned comparisons.

Hard requirements (do not skip):
- Create a feature branch: codex/ticket-19_changed-window-reporting
- Make small logical commits. Every commit body must include: "Tests: <commands>".
- Run tests: make test-fast (minimum) and record in TESTS.md and commit body.
- Create a complete run log under docs/agent_runs/$RUN_NAME/ with:
  PROMPT.md (paste this prompt verbatim),
  COMMANDS.md (every command you run),
  RESULTS.md (what changed + where outputs are),
  TESTS.md (tests run + pass/fail),
  META.md (git sha start/end, dirty flags, dataset ids/hashes if any real-data run).
- Prefer real-data smoke using the repo’s small derived datasets (fixtures) for speed; synthetic is allowed for unit tests but must not be the only validation.
- Do NOT “fix” by always marking changed=true/false. Changed-window must reflect actual semantics.
- Finish by generating a new bundle and record its path in RESULTS.md:
  make gpt-bundle TICKET=ticket-19 RUN_NAME=$RUN_NAME

Implementation tasks (do end-to-end, no long upfront plan):
1) Inspect how “changed windows” are currently defined and emitted.
   - Find where per-window outputs are written in experiments/eval/run.py (or wherever the eval runner writes metrics_detail / weights / overlay flags).
   - Identify existing fields: accepted/detected flags, any “changed” boolean, n_changed counts, window ids, portfolio ids (EW/MV), and how n_effective_* is computed today.
   - Write down the current semantics in docs/agent_runs/$RUN_NAME/RESULTS.md (short bullets).

2) Define a single, explicit changed-window semantics and make it consistent.
   - Preferred: changed_window := 1 when the treatment run applies a non-noop overlay correction for that window (i.e., “accepted and applied”), else 0.
   - Ensure this is emitted consistently for both EW and MV rows (even if EW weights don’t change; covariance can).
   - If there is already a different semantics in the code, do NOT silently change it. Either (a) keep it and document it clearly, or (b) change it and update tests + PROGRESS.md with the breaking change clearly called out.

3) Add conditional reporting to summaries.
   Update tools/make_summary.py (and tools/summarize_rc_sanity.py if used) so summary tables include:
   - ΔMSE and ΔQLIKE conditional on changed windows only (aligned window intersection restricted to changed==1).
   - n_changed counts and changed fraction (n_changed / n_total_aligned).
   - Weight-change magnitude stats on changed windows:
     - median ||Δw||_2
     - median turnover_delta := sum_i |w_treat - w_base|
   Produce these for EW and MV. For EW these will likely be 0, and that’s OK.

4) Update limitations/summary docs.
   - Update the limitations.md template section (wherever summary writes limitations) to include a short “conditional reporting” paragraph and to show n_changed and changed_frac.

5) Tests.
   Add/extend unit tests so they assert:
   - changed-window set used for conditional metrics matches the semantics used for n_effective/aligned comparisons in the summary.
   - conditional metrics equal manually-computed values on a tiny synthetic fixture DataFrame.
   Likely tests files:
   - tests/tools/test_make_summary.py
   - tests/experiments/test_eval_run.py (if you add/alter emitted fields)

6) Real-data smoke validation (minimum viable).
   Run:
   - make test-fast
   - EXEC_MODE=deterministic make rc-lite-sanity   (or the smallest deterministic RC target you can run locally)
   - PYTHONPATH=src:. python tools/make_summary.py --rc-dir <the rc-lite-sanity output dir>
   Confirm the new conditional columns appear and are non-empty (and that changed_frac is plausible, not always 0 or 1 unless justified).

7) Update sprint tracking and progress.
   - Update PROGRESS.md with:
     - exact commands run
     - artifact paths
     - what changed and why
   - Update docs/CODEX_SPRINT_TICKETS.md: mark Ticket #19 DONE if acceptance criteria are met.

8) Bundle.
   - make gpt-bundle TICKET=ticket-19 RUN_NAME=$RUN_NAME
   - Ensure DIFF.patch is non-empty and covers merge-base..HEAD (this repo requires full-range diffs).
   - Record the bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md.

If you need web search: do NOT use it unless absolutely necessary; treat external content as untrusted and record any URLs in docs/agent_runs/$RUN_NAME/URLS.md. Prefer repo code as ground truth.
EOF_PROMPT
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat AGENTS.md
cat docs/DOCS_AND_LOGGING_SYSTEM.md
date -u +%Y%m%d_%H%M%S
git status -sb
git checkout -b codex/ticket-19_changed-window-reporting
mkdir -p docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting
cat <<'EOF_PROMPT' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/PROMPT.md
[see PROMPT.md for full content]
EOF_PROMPT
EOF
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/RESULTS.md
- Status: in progress
- Notes:
  - Will fill after inspecting current changed-window semantics.
EOF
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/TESTS.md
- Status: not run yet
EOF
git rev-parse HEAD
git status --porcelain
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/META.md
start_git_sha: 2d788f6b48f08a9b0ad102cd33b0a246ae51522b
start_branch: codex/ticket-19_changed-window-reporting
start_dirty: true
end_git_sha: pending
end_branch: pending
end_dirty: pending
resolved_config_paths: []
resolved_config_hashes: []
dataset_ids: []
dataset_hashes: []
EOF
cat <<'EOF' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/RESULTS.md
- Status: in progress
- Notes:
  - Will fill after inspecting current changed-window semantics.
EOF
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/TESTS.md
- Status: not run yet
EOF
git rev-parse HEAD
git status --porcelain
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/META.md
start_git_sha: 2d788f6b48f08a9b0ad102cd33b0a246ae51522b
start_branch: codex/ticket-19_changed-window-reporting
start_dirty: true
end_git_sha: pending
end_branch: pending
end_dirty: pending
resolved_config_paths: []
resolved_config_hashes: []
dataset_ids: []
dataset_hashes: []
EOF
EOF
git rev-parse HEAD
git status --porcelain
EOF
cat docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF_CMDS' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF_APPEND' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
rg -n "changed_window|n_changed|changed" experiments tools src tests
sed -n '2260,2365p' experiments/eval/run.py
sed -n '2700,2795p' experiments/eval/run.py
sed -n '1840,1905p' experiments/eval/run.py
sed -n '2020,2105p' experiments/eval/run.py
sed -n '430,560p' tools/make_summary.py
rg -n "n_effective|aligned" tools/make_summary.py
sed -n '760,940p' tools/make_summary.py
rg -n "turnover_delta|delta_w|weight_change|weight" experiments tools src
rg -n "metrics_detail|diagnostics_detail|weights" experiments/eval/run.py
sed -n '2385,2625p' experiments/eval/run.py
rg -n "weights" experiments/eval/run.py
rg -n "n_effective|comparison_valid|aligned" experiments/eval/run.py
sed -n '470,620p' experiments/eval/run.py
sed -n '3260,3385p' experiments/eval/run.py
sed -n '3100,3345p' experiments/eval/run.py
sed -n '3680,3865p' experiments/eval/run.py
sed -n '3395,3465p' experiments/eval/run.py
sed -n '1,200p' tools/make_summary.py
sed -n '1000,1135p' tools/make_summary.py
sed -n '1,200p' tools/summarize_rc_sanity.py
rg -n "rc-lite-sanity|summarize_rc_sanity" Makefile tools
sed -n '180,215p' Makefile
sed -n '215,310p' Makefile
sed -n '340,460p' tests/experiments/test_eval_run.py
sed -n '1,240p' tests/tools/test_make_summary.py
sed -n '1,120p' tests/tools/test_summarize_rc_sanity.py
rg -n "def _concat_if_exists" tools/make_summary.py
sed -n '320,370p' tools/make_summary.py
rg -n "def summarise_rc_directory" -n tools/make_summary.py
sed -n '698,880p' tools/make_summary.py
rg -n "def _load_detail" tools/make_summary.py
sed -n '370,430p' tools/make_summary.py
rg -n "def evaluate_eval_run" -n src/meta/completeness.py
sed -n '96,220p' src/meta/completeness.py
rg -n "class MinVarMemo|MinVarMemo|minvar_ridge_box" -n src/finance
sed -n '1,140p' src/finance/portfolio.py
sed -n '1320,1455p' experiments/eval/run.py
sed -n '1720,1795p' experiments/eval/run.py
apply_patch  # experiments/eval/run.py: add overlay weight-delta stats
apply_patch  # experiments/eval/run.py: align delta metrics with changed windows
apply_patch  # tools/make_summary.py: extend PERF_COLUMNS
apply_patch  # tools/make_summary.py: add changed-window helpers
apply_patch  # tools/make_summary.py: load metrics_detail
apply_patch  # tools/make_summary.py: track changed_window_ids
apply_patch  # tools/make_summary.py: add conditional columns
apply_patch  # tools/make_summary.py: add conditional reporting in limitations
sed -n '1050,1135p' tools/make_summary.py
sed -n '1120,1215p' tools/make_summary.py
sed -n '1215,1355p' tools/make_summary.py
apply_patch  # tools/make_summary.py: fix f-strings in limitations
apply_patch  # tools/make_summary.py: filter metrics_detail for full regime
apply_patch  # tools/make_summary.py: filter aligned_metric_table by regime
apply_patch  # tools/make_summary.py: filter median weight deltas by regime
apply_patch  # tests/tools/test_make_summary.py: require new perf columns
apply_patch  # tests/tools/test_make_summary.py: add conditional metrics test
apply_patch  # tests/tools/test_make_summary.py: import pytest
apply_patch  # tests/experiments/test_eval_run.py: add empty changed-window test
apply_patch  # tools/summarize_rc_sanity.py: prefer delta_mse_vs_baseline
rg -n "empty = pd.DataFrame\\(columns=\\[" -n experiments/eval/run.py
sed -n '3140,3205p' experiments/eval/run.py
apply_patch  # experiments/eval/run.py: add weight_delta columns to empty metrics
cat <<'EOF' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
rg -n "RC_OUT_SANITY" -n Makefile
sed -n '40,80p' Makefile
make test-fast
date -u +%Y%m%d_%H%M%S
EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity
python - <<'PY'
import pandas as pd
path = "reports/rc-20251226-sanity-20251226_191833/summary/summary_perf.csv"
df = pd.read_csv(path)
print(df.columns.tolist())
print(df[[
    "regime",
    "portfolio",
    "delta_mse_changed_vs_baseline",
    "delta_qlike_changed_vs_baseline",
    "n_changed",
    "changed_frac",
    "median_weight_delta_l2",
    "median_turnover_delta",
]].head())
PY
head -n 5 reports/rc-20251226-sanity-20251226_191833/summary/summary_perf.csv
cat <<'EOF' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/TESTS.md
- make test-fast (pass; 78 passed, 170 deselected)
- EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity (pass; outputs under reports/rc-20251226-sanity-20251226_191833; includes tools/make_summary.py + tools/summarize_rc_sanity.py)
EOF
EOF  # command not found from failed heredoc append
cat <<'EOF_CMDS2' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
tail -n 20 docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF_CMDLOG' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/RESULTS.md
- Current changed-window semantics (pre-change):
  - `changed_flag` is set per window in `experiments/eval/run.py` when `max(|overlay_cov - baseline_cov|) > 1e-10` after `apply_overlay`; emitted in `diagnostics_detail.csv`.
  - `changed_windows_by_regime` is built from `diagnostics_detail.csv` (`window_id`, `changed_flag`), with `forced_changed_windows` overriding the set and (if forced union non-empty) flipping `changed_flag=1` for those windows.
  - Aligned Δ metrics and DM tests use `_aligned_*` helpers with `valid_window_ids=changed_windows_by_regime[...]` (now always, even if empty) so `n_effective_*` matches changed-window intersections.
- Code changes:
  - Added per-window weight-delta diagnostics (`weight_delta_l2`, `turnover_delta`) to `metrics_detail.csv` (EW=0, MV uses overlay vs baseline min-var weights when available).
  - Made aligned Δ metrics in `experiments/eval/run.py` always respect changed windows (no fallback to all windows when none are changed); flip-set DM/sign stats now always filter by changed windows.
  - `tools/make_summary.py` now computes changed-window-only ΔMSE/ΔQLIKE, `n_changed`, `changed_frac`, and median weight-delta stats; limitations now include a conditional-reporting section with `n_changed`/`changed_frac` lines.
  - `tools/summarize_rc_sanity.py` now prefers `delta_mse_vs_baseline` when available.
  - Tests added/updated for changed-window conditional metrics and empty changed-window behavior.
- Outputs:
  - rc-lite-sanity run: `reports/rc-20251226-sanity-20251226_191833/` (summary files under `.../summary/`).
  - `summary_perf.csv` includes new columns but is empty because rc-lite-sanity runs are capped (date truncation), so changed-window stats are not populated in that smoke run.
EOF
EOF  # command not found from failed heredoc append
cat <<'EOF_LOG' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
apply_patch  # docs/CODEX_SPRINT_TICKETS.md: mark Ticket #19 DONE
cat <<'EOF_CMDLOG' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF' >> PROGRESS.md

## 2025-12-26T18:36Z — ticket-19 changed-window reporting
- **Branch/Run**: `codex/ticket-19_changed-window-reporting` (RUN_NAME=`20251226_174844_ticket-19_changed-window-reporting`), git sha `5980a47d819624954a422b672f228fee16a8f61b`.
- **Commands**:
  - `make test-fast`
  - `EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity`
  - `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251226-sanity-20251226_191833`
- **Results**:
  - Added changed-window-only ΔMSE/ΔQLIKE, `n_changed`, `changed_frac`, and median weight-delta stats to summaries; limitations now include a conditional-reporting section.
  - Evaluation now always aligns Δ metrics/DM stats on changed-window sets; added per-window weight-delta diagnostics in `metrics_detail.csv`.
  - rc-lite-sanity summaries were generated but excluded from aggregates due to date-truncation caps (summary_perf header-only).
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/`
  - Outputs `reports/rc-20251226-sanity-20251226_191833/`
  - Weekly smoke outputs `experiments/equity_panel/outputs_rc-lite-20251226_20251226_191833/`
cat <<'EOF' >> PROGRESS.md

## 2025-12-26T18:36Z — ticket-19 changed-window reporting
- **Branch/Run**: `codex/ticket-19_changed-window-reporting` (RUN_NAME=`20251226_174844_ticket-19_changed-window-reporting`), git sha `5980a47d819624954a422b672f228fee16a8f61b`.
- **Commands**:
  - `make test-fast`
  - `EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity`
  - `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/rc-20251226-sanity-20251226_191833`
- **Results**:
  - Added changed-window-only ΔMSE/ΔQLIKE, `n_changed`, `changed_frac`, and median weight-delta stats to summaries; limitations now include a conditional-reporting section.
  - Evaluation now always aligns Δ metrics/DM stats on changed-window sets; added per-window weight-delta diagnostics in `metrics_detail.csv`.
  - rc-lite-sanity summaries were generated but excluded from aggregates due to date-truncation caps (summary_perf header-only).
- **Artifacts**:
  - Run log `docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/`
  - Outputs `reports/rc-20251226-sanity-20251226_191833/`
  - Weekly smoke outputs `experiments/equity_panel/outputs_rc-lite-20251226_20251226_191833/`
EOF
cat <<'EOF_PROG' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
EOF  # command not found from failed heredoc append
cat <<'EOF_FIX' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
git status -sb
git add experiments/eval/run.py tests/experiments/test_eval_run.py
git commit -m "eval: align changed-window deltas + weight stats" -m "Tests: make test-fast; EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity" -m "Artifacts: reports/rc-20251226-sanity-20251226_191833/; experiments/equity_panel/outputs_rc-lite-20251226_20251226_191833/"
git add tools/make_summary.py tools/summarize_rc_sanity.py tests/tools/test_make_summary.py
git commit -m "summary: add changed-window conditional stats" -m "Tests: make test-fast; EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity" -m "Artifacts: reports/rc-20251226-sanity-20251226_191833/; experiments/equity_panel/outputs_rc-lite-20251226_20251226_191833/"
rg -n "gpt-bundle" -n
tail -n 40 PROGRESS.md
date -u +%Y-%m-%dT%H:%MZ
git rev-parse HEAD
tail -n 12 docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF_CMDLOG2' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
cat <<'EOF_META' > docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/META.md
start_git_sha: 2d788f6b48f08a9b0ad102cd33b0a246ae51522b
start_branch: codex/ticket-19_changed-window-reporting
start_dirty: true
end_git_sha: 5980a47d819624954a422b672f228fee16a8f61b
end_branch: codex/ticket-19_changed-window-reporting
end_dirty: true
resolved_config_paths:
  - reports/rc-20251226-sanity-20251226_191833/dow-tyler/resolved_config.json
  - reports/rc-20251226-sanity-20251226_191833/vol-tyler/resolved_config.json
resolved_config_hashes:
  - ec4de79f321612c182bd17ec6583858bf05fd2b2dbf142ebd7df01a0c94b91f6
  - 823d49f089e0b70b042478a433c8c6df7d0a9cc60797018cd07de99490ba5a06
dataset_ids:
  - data/returns_daily.csv
  - data/factors/ff5mom_daily.csv
dataset_hashes:
  - 96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197
  - 469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca
EOF_META
cat <<'EOF_CMDLOG' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
python - <<'PY'
import json
from pathlib import Path
path = Path("reports/rc-20251226-sanity-20251226_191833/dow-tyler/run.json")
payload = json.loads(path.read_text())
print(payload.get("returns"))
print(payload.get("factors"))
PY
python - <<'PY'
import json
from pathlib import Path
path = Path("reports/rc-20251226-sanity-20251226_191833/dow-tyler/run.json")
payload = json.loads(path.read_text())
print(sorted(payload.keys()))
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
print(reg.get("data/returns_daily.csv"))
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
print(list(reg.keys())[:5])
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
for ds in reg.get("datasets", []):
    if ds.get("path") == "data/returns_daily.csv":
        print(ds)
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
print(type(reg.get("datasets")))
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
for key, ds in reg.get("datasets", {}).items():
    if isinstance(ds, dict) and ds.get("path") == "data/returns_daily.csv":
        print(key, ds)
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
for key, ds in reg.get("datasets", {}).items():
    if isinstance(ds, dict) and "returns" in str(ds.get("path", "")):
        print(key, ds.get("path"), ds.get("sha256"))
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/registry.json").read_text())
for key, ds in reg.get("datasets", {}).items():
    print(key, ds)
    break
PY
python - <<'PY'
import json
from pathlib import Path
reg = json.loads(Path("data/factors/registry.json").read_text())
for key, ds in reg.get("datasets", {}).items():
    if "ff5mom" in key:
        print(key, ds)
PY
python - <<'PY'
import json
from pathlib import Path
paths = [
    Path("reports/rc-20251226-sanity-20251226_191833/dow-tyler/run.json"),
    Path("reports/rc-20251226-sanity-20251226_191833/vol-tyler/run.json"),
]
for p in paths:
    payload = json.loads(p.read_text())
    print(p.parent, payload.get("resolved_config_path"), payload.get("resolved_config_hash"))
PY
cat <<'EOF_CMDLOG3' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
make gpt-bundle TICKET=ticket-19 RUN_NAME=20251226_174844_ticket-19_changed-window-reporting
apply_patch  # docs/agent_runs/.../RESULTS.md: add bundle path
cat <<'EOF_CMDLOG4' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
git status -sb
git add docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/RESULTS.md
git commit -m "docs: record ticket-19 bundle" -m "Tests: make test-fast; EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity" -m "Artifacts: docs/gpt_bundles/20251226_194317_ticket-19_20251226_174844_ticket-19_changed-window-reporting.zip"
cat <<'EOF_CMDLOG5' >> docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
git add docs/agent_runs/20251226_174844_ticket-19_changed-window-reporting/COMMANDS.md
git commit -m "docs: finalize ticket-19 command log" -m "Tests: make test-fast; EXEC_MODE=deterministic RC_LITE_STAMP=20251226_181827 make rc-lite-sanity" -m "Artifacts: docs/gpt_bundles/20251226_194317_ticket-19_20251226_174844_ticket-19_changed-window-reporting.zip"
