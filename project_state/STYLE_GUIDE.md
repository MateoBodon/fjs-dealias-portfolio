---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Style Guide

- **Formatting** — Black (line length 88) + Ruff; run `make fmt` before commits. Type hints preferred on public functions; Python ≥3.11 features allowed.
- **Imports** — Prefer absolute `src.`/package imports over deep relatives; keep experiment scripts thin and delegate to `src/` modules.
- **Configs** — Avoid hard-coding experiment parameters inside `src/`; prefer YAML in `experiments/**/config*.yaml` and CLI flags. Log resolved configs (`resolved_config.json` / `config_resolved.yaml`).
- **Data handling** — Never commit WRDS/raw data. Use `tools/verify_dataset.py` and registries to validate hashes. Write new outputs to timestamped directories; do not delete prior RC drops.
- **Logging** — Ensure runs emit `run.json`/`resolved_config.*` and completeness metadata. For new guardrails, add diagnostics columns rather than silent skips.
- **Docs** — Keep `project_state/` concise, path-centric, and dated. When adding public knobs, update `CONFIG_REFERENCE.md` + PROGRESS entry.
- **Tests** — Prefer `make test-fast` before commits. When changing detection/overlay or evaluation logic, add targeted tests (smoke + edge cases) and note markers.
