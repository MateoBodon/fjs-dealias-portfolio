# Commands

```
ls
ls src
sed -n '1,200p' Makefile
ls project_state
ls project_state/_generated
git status -sb
ls docs/agent_runs | tail -n 5
sed -n '1,200p' PROGRESS.md
sed -n '1,200p' docs/DOCS_AND_LOGGING_SYSTEM.md
python3 --version
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
date -u +"%Y-%m-%dT%H:%M:%SZ"
uname -a
sed -n '1,200p' pyproject.toml
ls tools | rg -n "project_state|generate"
sed -n '1,200p' tools/generate_project_state.py
sed -n '200,400p' tools/generate_project_state.py
apply_patch (update tools/generate_project_state.py)
python3 tools/generate_project_state.py
head -n 20 project_state/_generated/symbol_index.json
head -n 20 project_state/_generated/import_graph.json
sed -n '1,120p' project_state/ARCHITECTURE.md
sed -n '1,200p' project_state/CONFIG_REFERENCE.md
sed -n '1,200p' project_state/MODULE_SUMMARIES.md
rg --files src | head
rg --files src/finance | head
rg --files src/finance | rg portfolios
ls experiments/equity_panel
sed -n '1,200p' project_state/EXPERIMENTS.md
sed -n '1,200p' project_state/DATAFLOW.md
sed -n '1,200p' project_state/RESEARCH_NOTES.md
sed -n '1,200p' project_state/OPEN_QUESTIONS.md
sed -n '1,200p' project_state/KNOWN_ISSUES.md
sed -n '1,200p' project_state/STYLE_GUIDE.md
sed -n '1,200p' project_state/TEST_COVERAGE.md
sed -n '1,200p' project_state/SERVER_ENVIRONMENT.md
sed -n '1,200p' project_state/CHANGELOG.md
sed -n '1,120p' project_state/INDEX.md
sed -n '1,200p' project_state/CURRENT_RESULTS.md
sed -n '1,200p' docs/agent_runs/20251222_183526_ticket-07_advisor-ready-dow/RESULTS.md
sed -n '1,200p' docs/agent_runs/20251222_014730_ticket-06_window-coverage/RESULTS.md
sed -n '1,200p' docs/agent_runs/20251221_220252_ticket-05_advisor-ready-rc/RESULTS.md
ls REPO_PLAN.md
ls docs/PLAN_OF_RECORD.md
rg --files -g 'config.paper*'
ls experiments/eval
rg -n "mv_solv" experiments/eval/run.py
rg -n "add_argument" experiments/equity_panel/run.py | head -n 40
sed -n '3810,3890p' experiments/equity_panel/run.py
sed -n '3890,3970p' experiments/equity_panel/run.py
sed -n '3970,4055p' experiments/equity_panel/run.py
sed -n '4055,4140p' experiments/equity_panel/run.py
sed -n '1,120p' src/data/panels.py
sed -n '1,120p' src/data/registry.py
rg -n "config.paper_v1|paper_v1|config" experiments/eval/config.py
sed -n '100,150p' experiments/eval/config.py
rg -n "import_graph.json"
python3 - <<'PY' (list module names by top-level)
python3 - <<'PY' (find empty module names)
apply_patch (update module_name_from_path)
python3 tools/generate_project_state.py
python3 - <<'PY' (list module names by top-level)
head -n 20 project_state/_generated/import_graph.json
date -u +"%Y%m%d_%H%M%S"
git rev-parse --short HEAD
git checkout -b chore/project_state_refresh
git checkout chore/project_state_refresh
python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
sed -n '1,40p' project_state/FUNCTION_INDEX.md
sed -n '1,200p' project_state/INDEX.md
python3 - <<'PY' (rewrite INDEX.md)
python3 - <<'PY' (write project_state docs; first attempt failed)
python3 - <<'PY' (write project_state docs; second attempt succeeded)
sed -n '1,60p' project_state/ARCHITECTURE.md
git rev-parse HEAD
sed -n '1,120p' project_state/MODULE_SUMMARIES.md
sed -n '1,160p' project_state/CURRENT_RESULTS.md
sed -n '1,160p' project_state/CONFIG_REFERENCE.md
sed -n '1,40p' project_state/INDEX.md
sed -n '1,12p' project_state/FUNCTION_INDEX.md
sed -n '1,12p' project_state/DEPENDENCY_GRAPH.md
apply_patch (update tools/generate_project_state.py docstring)
python3 tools/generate_project_state.py
python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
ls docs/agent_runs | tail -n 5
ls docs/agent_runs | sort | tail -n 5
sed -n '1,200p' docs/agent_runs/20251220_231000_rc-lite-deterministic/RESULTS.md
sed -n '1,200p' docs/agent_runs/20251220_223706_ticket-09_weekly-gating-attribution/RESULTS.md
ls src/fjs
ls src/data
ls tools/__init__.py
RUN_NAME=20251222_205400_project_state_rebuild; mkdir -p docs/agent_runs/$RUN_NAME
cat <<'EOF' > docs/agent_runs/$RUN_NAME/PROMPT.md ...
cat <<'EOF' > docs/agent_runs/$RUN_NAME/TESTS.md ...
python3 - <<'PY' (write META.json + META.md)
STAMP=20251222_205400; SHORT=$(git rev-parse --short HEAD); zip -r docs/gpt_bundles/project_state_${STAMP}_${SHORT}.zip project_state PROGRESS.md docs/PLAN_OF_RECORD.md docs/DOCS_AND_LOGGING_SYSTEM.md
python3 - <<'PY' (prepend PROGRESS.md entry)
cat <<'EOF' > docs/agent_runs/$RUN_NAME/RESULTS.md ...
git add project_state tools/generate_project_state.py PROGRESS.md docs/agent_runs/20251222_205400_project_state_rebuild
git commit -m "Rebuild project_state @ a7d76d8" -m "Tests: not run (doc-only)"
```
