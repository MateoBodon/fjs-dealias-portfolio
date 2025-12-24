ls
cat AGENTS.md
date -u +%Y%m%d_%H%M%S
RUN_NAME=20251223_222840_ticket-18_inject-spike-sensitivity
mkdir -p docs/agent_runs/$RUN_NAME
: > docs/agent_runs/$RUN_NAME/PROMPT.md
: > docs/agent_runs/$RUN_NAME/COMMANDS.md
: > docs/agent_runs/$RUN_NAME/RESULTS.md
: > docs/agent_runs/$RUN_NAME/TESTS.md
: > docs/agent_runs/$RUN_NAME/META.md
cat <<'EOF' > docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/COMMANDS.md
cat <<'EOF' > docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/PROMPT.md
git status -sb
git checkout -b codex/ticket-18-inject-spike-sensitivity
git rev-parse HEAD
date -u +%Y-%m-%dT%H:%M:%SZ
cat <<'EOF' > docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/META.md
rg -n "inject-spike" Makefile
sed -n '480,560p' Makefile
sed -n '1,240p' experiments/eval/inject_spike.py
sed -n '240,520p' experiments/eval/inject_spike.py
python experiments/eval/inject_spike.py --help
PYTHONPATH=src:. python experiments/eval/inject_spike.py --help
rg -n "def detect_spikes|class Overlay" -n src/fjs/overlay.py
sed -n '1,220p' src/fjs/overlay.py
rg -n "def detect_spikes|def apply_overlay" -n src/fjs/overlay.py
sed -n '300,460p' src/fjs/overlay.py
rg -n "serialise|resolve_eval_config" -n experiments/eval -g"*.py"
sed -n '1,200p' experiments/eval/config.py
sed -n '320,420p' experiments/eval/run.py
rg -n "dataset|hash|id" experiments/eval/run.py
sed -n '180,260p' experiments/eval/run.py
sed -n '420,520p' experiments/eval/run.py
rg -n "EvalConfig\\(" -n experiments/eval/config.py
sed -n '340,420p' experiments/eval/config.py
rg -n "run.json" -n experiments/eval/run.py
sed -n '3160,3245p' experiments/eval/run.py
rg -n "inject_spike|spike" tests
cat pytest.ini
rg -n "test-fast" -n Makefile
sed -n '1,60p' Makefile
rg -n "RC_START|RC_END|RC_WINDOW|RC_HORIZON" Makefile
rg -n "ASSETS_TOP" Makefile
sed -n '80,130p' Makefile
rg -n "RC_INJECT_OUT" -n Makefile
sed -n '320,350p' Makefile
rg -n "class Factor|factor_entry" -n experiments/eval/run.py
rg -n "class Prewhiten|mode_effective" -n experiments/eval/run.py
sed -n '320,520p' experiments/eval/inject_spike.py
cat <<'EOF' > tests/experiments/test_inject_spike.py
make test-fast
RC_ASSETS_TOP=50 RC_WINDOW=60 RC_HORIZON=10 RC_START=2024-01-01 RC_END=2024-03-31 make inject-spike
RC_ASSETS_TOP=50 RC_WINDOW=60 RC_HORIZON=10 RC_START=2024-01-01 RC_END=2024-06-30 make inject-spike
PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-03-31 --assets-top 30 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3 --seed 7 --out reports/inject_spike
ls -1t reports/inject_spike | head -n 5
for d in 20251223_224954 20251223_224638 20251223_224624; do echo "--- "; ls -1 reports/inject_spike/; done
PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-03-31 --assets-top 30 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3,6,9 --seed 7 --out reports/inject_spike
ls -1t reports/inject_spike | head -n 5
cat reports/inject_spike/20251223_224954/run.json
cat reports/inject_spike/20251223_225141/run.json
python - <<'PY' (write run.json for aborted inject_spike runs)
cat docs/CODEX_SPRINT_TICKETS.md
cat PROGRESS.md
cat project_state/RESEARCH_NOTES.md
git status -sb
cat .gitignore
ls -1t reports/inject_spike | head -n 5
python -m py_compile experiments/eval/inject_spike.py
git add experiments/eval/inject_spike.py Makefile
git commit -m "Add inject-spike curve outputs" -m "Tests: make test-fast"
git add tests/experiments/test_inject_spike.py
git commit -m "Add inject-spike unit tests" -m "Tests: make test-fast"
git rev-parse HEAD
python - <<'PY' (reorder ticket-18 block in PROGRESS.md)
sed -n '1,30p' PROGRESS.md
date -u +%Y-%m-%dT%H:%M:%SZ
python - <<'PY' (update RESEARCH_NOTES.md front matter)
sed -n '1,20p' project_state/RESEARCH_NOTES.md
git add PROGRESS.md docs/CODEX_SPRINT_TICKETS.md project_state/RESEARCH_NOTES.md docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity reports/inject_spike
git commit -m "Document injection sensitivity run" -m "Tests: make test-fast"
git add docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/COMMANDS.md
git commit -m "Update run log commands" -m "Tests: make test-fast"
make gpt-bundle TICKET=ticket-18 RUN_NAME=20251223_222840_ticket-18_inject-spike-sensitivity
git add docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/COMMANDS.md docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/RESULTS.md
git commit -m "Finalize bundle log" -m "Tests: make test-fast"
PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-09-30 --assets-top 30 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3,6,9,12,15 --seed 7 --out reports/inject_spike
ls -1t reports/inject_spike | head -n 5
ls -1 reports/inject_spike/20251224_051229
python - <<'PY' (write run.json for aborted inject_spike 20251224_051229)
PYTHONPATH=src:. python experiments/eval/inject_spike.py --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --window 40 --horizon 5 --start 2024-01-01 --end 2024-06-30 --assets-top 25 --config experiments/eval/config.yaml --thresholds experiments/eval/thresholds.json --group-design week --use-factor-prewhiten 1 --mu-grid 3,6,9,12,15 --seed 7 --out reports/inject_spike
ls -1t reports/inject_spike | head -n 5
cat reports/inject_spike/20251224_051700/run.json
cat reports/inject_spike/20251224_051700/curve.csv
git add PROGRESS.md project_state/RESEARCH_NOTES.md docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/COMMANDS.md docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/RESULTS.md reports/inject_spike/20251224_051229 reports/inject_spike/20251224_051700
git commit -m "Record larger injection run" -m "Tests: not run (not requested)"
git rev-parse HEAD
date -u +%Y-%m-%dT%H:%M:%SZ
python - <<'PY' (update RESEARCH_NOTES.md front matter after new run)
git add project_state/RESEARCH_NOTES.md docs/agent_runs/20251223_222840_ticket-18_inject-spike-sensitivity/COMMANDS.md
git commit -m "Update injection notes metadata" -m "Tests: not run (not requested)"
make gpt-bundle TICKET=ticket-18 RUN_NAME=20251223_222840_ticket-18_inject-spike-sensitivity
