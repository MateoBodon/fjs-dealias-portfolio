# Commands

git status --porcelain

date +%Y%m%d_%H%M%S

mkdir -p docs/agent_runs/20251221_065852_ticket-00_data-mount

cat > docs/agent_runs/20251221_065852_ticket-00_data-mount/PROMPT.md <<'PROMPT_EOF'
...prompt written...
PROMPT_EOF

git rev-parse HEAD

ls -lah /Volumes/Storage/Data

apply_patch (docs/DOCS_AND_LOGGING_SYSTEM.md)

apply_patch (PROGRESS.md)

cat > docs/agent_runs/20251221_065852_ticket-00_data-mount/RESULTS.md <<'EOF'
...updated RESULTS.md...
EOF

cat > docs/agent_runs/20251221_065852_ticket-00_data-mount/META.md <<'EOF'
...written META.md...
EOF

cat > docs/agent_runs/20251221_065852_ticket-00_data-mount/TESTS.md <<'EOF'
...written TESTS.md...
EOF

git add docs/DOCS_AND_LOGGING_SYSTEM.md PROGRESS.md docs/agent_runs/20251221_065852_ticket-00_data-mount

git commit -m "Document external WRDS data mirror location" -m "Tests run: not run (doc-only)"

git diff 8ac9c14cd0e134aee8bf7a803891a0c476250a95..HEAD > docs/agent_runs/20251221_065852_ticket-00_data-mount/DIFF.patch

make gpt-bundle TICKET=data-mount RUN_NAME=20251221_065852_ticket-00_data-mount

git commit -m "Update data-mount run log artifacts" -m "Tests run: not run (log-only)"

git commit -m "Record data-mount bundle output" -m "Tests run: not run (bundle logging)"

make gpt-bundle TICKET=data-mount RUN_NAME=20251221_065852_ticket-00_data-mount
