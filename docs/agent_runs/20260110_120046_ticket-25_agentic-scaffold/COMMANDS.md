# Commands

```text
python3 /root/.codex/skills/repo-bootstrap/scripts/bootstrap_repo.py --force
sed -n '1,200p' PROJECT.md
ls
sed -n '1,200p' README.md
sed -n '1,200p' AGENTS.md
sed -n '1,200p' Makefile
rg --files -g 'Makefile' -g 'pyproject.toml' -g 'ruff.toml' -g 'mypy.ini' -g 'pytest.ini'
sed -n '1,200p' pyproject.toml
sed -n '1,200p' ruff.toml
sed -n '1,200p' mypy.ini
sed -n '1,200p' Makefile.bak.20260110_115545
ls src
sed -n '1,200p' .gitignore
sed -n '1,200p' .gitignore.append
sed -n '1,200p' PROGRESS.md
sed -n '1,200p' docs/DOCS_AND_LOGGING_SYSTEM.md
sed -n '1,200p' docs/CODEX_SPRINT_TICKETS.md
date +%Y%m%d_%H%M%S
cp Makefile.bak.20260110_115545 Makefile
tail -n 40 Makefile
rg -n "project-state|snapshot|bundle|gpt-bundle" Makefile
rg -n "project-state|project_state|snapshot|repo_snapshot" Makefile
apply_patch (update AGENTS.md)
apply_patch (update PROJECT.md)
apply_patch (update docs/CODEX_SPRINT_TICKETS.md)
apply_patch (update docs/DECISIONS.md)
python3 tools/agentic/project_state_refresh.py --zip
make test-fast
date +%Y%m%d_%H%M%S
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git status --porcelain
mkdir -p docs/agent_runs/20260110_120046_ticket-25_agentic-scaffold
cat <<'EOF' > docs/agent_runs/20260110_120046_ticket-25_agentic-scaffold/PROMPT.md
cat <<'EOF' > docs/agent_runs/20260110_120046_ticket-25_agentic-scaffold/RESULTS.md
cat <<'EOF' > docs/agent_runs/20260110_120046_ticket-25_agentic-scaffold/TESTS.md
cat <<'EOF' > docs/agent_runs/20260110_120046_ticket-25_agentic-scaffold/META.md
apply_patch (update PROGRESS.md)
cat <<'EOF' > docs/agent_runs/20260110_120046_ticket-25_agentic-scaffold/COMMANDS.md
```
