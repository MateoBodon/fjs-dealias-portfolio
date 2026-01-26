# Agent Run Logs

Run logs live under `docs/agent_runs/<RUN_NAME>/` and are tracked.

Bundling notes:
- GPT bundles are emitted to `artifacts/_local/gpt_bundles/` (scratch, ignored).
- Bundling is allowed on dirty trees; the bundler will temporarily stash and restore changes (or can be run with `--no-stash` to require a clean tree).
