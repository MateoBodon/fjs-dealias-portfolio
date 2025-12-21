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
