# Commands

1. ls
2. cat AGENTS.md
3. cat PROJECT.md
4. cat PROGRESS.md
5. rg -n "DOCS_AND_LOGGING_SYSTEM" -n docs/DOCS_AND_LOGGING_SYSTEM.md && cat docs/DOCS_AND_LOGGING_SYSTEM.md
6. cat docs/CODEX_SPRINT_TICKETS.md
7. rg --files -g 'AGENTS.md'
8. ls project_state
9. cat project_state/README.md
10. cat project_state/ARCHITECTURE.md
11. cat project_state/RUNBOOK.md
12. cat project_state/CURRENT_RESULTS.md
13. cat project_state/KNOWN_ISSUES.md
14. cat project_state/INDEX.md
15. cat project_state/CHANGELOG.md
16. cat project_state/ROADMAP.md
17. cat project_state/OPEN_QUESTIONS.md
18. cat project_state/RESEARCH_NOTES.md
19. cat project_state/DATAFLOW.md
20. cat project_state/PIPELINE_FLOW.md
21. cat RUNBOOK.md
22. rg -n "rc-lite|rc-lite-sanity|rc-week|rc-dow|inject-spike|gpt-bundle|test-fast" Makefile
23. sed -n '1,140p' Makefile
24. sed -n '180,260p' Makefile
25. sed -n '260,360p' Makefile
26. python3 - <<'PY'\nimport inspect, textwrap\nfrom pathlib import Path\npath = Path('tools/agentic/project_state_refresh.py')\nprint(path)\nprint(path.read_text())\nPY
27. date +%Y%m%d_%H%M%S
28. git rev-parse HEAD
29. git rev-parse --abbrev-ref HEAD
30. git status --porcelain=v1 -b
31. git checkout -b codex/ticket-26-project-state-refresh
32. mkdir -p docs/agent_runs/20260126_010603_ticket-26_project-state-refresh
33. cat <<'EOF' > docs/agent_runs/20260126_010603_ticket-26_project-state-refresh/PROMPT.md ...
34. cat <<'EOF' > docs/agent_runs/20260126_010603_ticket-26_project-state-refresh/COMMANDS.md ...
35. date -u +%Y-%m-%dT%H:%M:%SZ
36. apply_patch (update docs/CODEX_SPRINT_TICKETS.md)
37. apply_patch (update project_state/RUNBOOK.md)
38. apply_patch (update project_state/ARCHITECTURE.md)
39. apply_patch (update project_state/CURRENT_RESULTS.md)
40. apply_patch (update project_state/KNOWN_ISSUES.md)
41. apply_patch (update project_state/ROADMAP.md)
42. apply_patch (update project_state/OPEN_QUESTIONS.md)
43. apply_patch (update project_state/RESEARCH_NOTES.md)
44. apply_patch (update project_state/BACKLOG.md)
45. apply_patch (update project_state/CHANGELOG.md)
46. apply_patch (update project_state/DATAFLOW.md)
47. apply_patch (update project_state/PIPELINE_FLOW.md)
48. cat project_state/EXPERIMENTS.md
49. apply_patch (update project_state/EXPERIMENTS.md)
50. python3 tools/agentic/project_state_refresh.py --zip
51. make test-fast
52. git status --porcelain=v1
53. git status --porcelain=v1 project_state/_generated
54. cat <<'EOF' > docs/agent_runs/20260126_010603_ticket-26_project-state-refresh/RESULTS.md ...
55. cat <<'EOF' > docs/agent_runs/20260126_010603_ticket-26_project-state-refresh/TESTS.md ...
56. cat <<'EOF' > docs/agent_runs/20260126_010603_ticket-26_project-state-refresh/META.md ...
57. apply_patch (update PROGRESS.md)
58. git status --porcelain=v1
