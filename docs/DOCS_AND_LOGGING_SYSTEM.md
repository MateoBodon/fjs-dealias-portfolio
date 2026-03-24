# DOCS_AND_LOGGING_SYSTEM.md (enforced protocol)

This file defines the documentation + logging contract for this repo.
If you violate it, your results are not mergeable and not citeable.

---

## 1) Canonical directories (where things go)

Repo root:
- `AGENTS.md` — stop-the-line rules for humans + agents (this is enforced)
- `PROGRESS.md` — chronological log of what changed + what was run (**required update per merged ticket**)

Documentation:
- `docs/PLAN_OF_RECORD.md` — research framing + acceptance criteria + roadmaps
- `docs/DOCS_AND_LOGGING_SYSTEM.md` — this file (logging contract)
- `docs/CODEX_SPRINT_TICKETS.md` — next sprint tickets (ordered, scoped)

Prompts / agent traces:
- `docs/prompts/` — prompt text used for significant runs (one file per run; optional if PROMPT.md exists in agent run log)
- `docs/gpt_outputs/` — raw GPT outputs (Prompt-1/2/3 diagnostics etc), immutable
- `docs/agent_runs/<RUN_NAME>/` — **required** run log folder per Codex run (details below)
- `artifacts/_local/gpt_bundles/` — zip bundles produced by `make gpt-bundle` (scratch)

Experiment outputs:
- `reports/` — daily evaluation + summary artifacts
- `experiments/equity_panel/outputs_*` — weekly runner outputs
- `.cache/` — cached panels/per-window stats (**never** “source of truth”)

---

## 2) Run naming (one scheme, everywhere)

RUN_NAME format:
- `<YYYYMMDD_HHMMSS>_ticket-<NN>_<short-slug>`
Examples:
- `20251223_091500_ticket-16_paper-config-integrity`
- `20251223_154210_ticket-18_inject-spike-real-windows`

Rules:
- timestamp: local or UTC is fine, but consistent within a sprint
- slug: kebab-case describing the change (NOT the result)
- ticket number: must match `docs/CODEX_SPRINT_TICKETS.md`

---

## 3) Required contents of `docs/agent_runs/<RUN_NAME>/`

Every Codex run MUST create these files:

- `PROMPT.md`
  - exact prompt text given to Codex (verbatim)
- `COMMANDS.md`
  - every command executed (including tests), in order
  - include environment variables that affect reproducibility (EXEC_MODE, thread caps, etc.)
- `RESULTS.md`
  - what changed (bullets)
  - links to artifact directories (reports/*, experiments/*)
  - key finding(s) and any surprises
  - any failures + how they were resolved
- `TESTS.md`
  - exact tests/commands executed
  - pass/fail summary
- `META.json` (canonical)
  - git SHA before/after
  - branch name
  - whether repo was dirty at start
  - resolved config path(s) used
  - config hash(es) (sha256 of resolved_config.*)
  - dataset ids/hashes used (from registry + verify step)

Legacy compatibility:
- `META.md` may exist for historical runs, but new runs must write `META.json`.

Recommended (strongly):
- `DIFF.patch` — `git show --patch --stat --binary <REV>` (default `HEAD`) saved for fast review
- `bundle_contents.txt` — if you ran `make gpt-bundle`, capture `unzip -l ...`
- `URLS.md` — if web search was enabled, list every URL consulted (treat as untrusted)

---

## 4) Experiment run metadata (must exist inside output dirs)

For any run that writes to `reports/` or `experiments/.../outputs_*`, the output dir must contain:

- `run.json` or `run_manifest.json` (preferred: `run.json`)
  - must include:
    - `git_sha`, `git_dirty`
    - `cap_active`, `cap_sources`
    - dataset paths + sha256 ids (from registries)
    - resolved config path + config hash
    - key knobs (design, p, window/horizon, edge_mode, gate params, shrinker, prewhiten flag)
    - portfolio knobs (ridge/box/turnover/condition cap, solver name/status, skip policy)
    - skip counts by reason
- `resolved_config.json` (daily) or `config_resolved.yaml` (weekly)
  - exact final resolved config written to disk
- failures must be recorded:
  - skip counts/shares by reason
  - exception class + stage + minimal context (no opaque “failure” bucket)

---

## 5) Update rules (what MUST be updated per merged ticket)

Per merged ticket, you MUST update:
- `PROGRESS.md` (one entry with date, branch, SHAs, commands, tests, and artifacts)
- If a run log’s `COMMANDS.md` records multiple `BUNDLE_STAMP=` values, `PROGRESS.md` must cite the final stamp bundle path for that `RUN_NAME` and mark earlier bundle paths as superseded.

If results changed materially:
- `project_state/CURRENT_RESULTS.md`

If a blocker is fixed or discovered:
- `project_state/KNOWN_ISSUES.md`

If behavior/config knobs changed:
- `project_state/CONFIG_REFERENCE.md`

If tests/targets changed:
- `project_state/TEST_COVERAGE.md` and (if applicable) Makefile notes

---

## 6) “Validated run” labeling (no contamination)

A run can be labeled **validated** only if:
- deterministic mode where applicable (`EXEC_MODE=deterministic` + thread caps)
- `cap_active=false` (or explicitly labeled as *not headline*)
- no silent fallbacks (config/solver/window dropping)
- skip/guard reasons are attributable (no opaque buckets)
- summary tables clearly state:
  - effective sample size (`n_effective_*`) used for Δ/DM
  - skip rates and whether comparisons are aligned intersections

Policy:
- If a run is capped, summary tooling must segregate it (separate section) and it cannot be used for headline claims.

---

## 7) Bundling for GPT review / advisor audit

After each merged ticket (or at least once per sprint), run:
- `make gpt-bundle TICKET=<ticket-id> RUN_NAME=<RUN_NAME>`

Bundle MUST include:
- `AGENTS.md`, `PROGRESS.md`, `docs/*`, `project_state/*`
- `DIFF.patch` (generated from `git diff --binary <merge-base>..HEAD`) and `LAST_COMMIT.txt`
- `BUNDLE_META.md` (base/head metadata for the diff range, including `git_dirty`)
- final markdown snapshots for files changed in the diff range (for example `docs/gpt_outputs/*`, `docs/prompts/*`, `docs/tickets/*`) so reviewers can inspect full file state, not only patch hunks
- the run log folder under `docs/agent_runs/<RUN_NAME>/`
- (if applicable) key outputs under `reports/` or `experiments/.../outputs_*`

Bundling fails loud (non-zero) if:
- `DIFF.patch` would be empty
- base ref cannot be resolved (set `BUNDLE_BASE` to override)
- required run log files are missing (`PROMPT.md`, `COMMANDS.md`, `RESULTS.md`, `TESTS.md`, `META.json`)
- for runs with timestamped names `>= 20260216_000000`: if `COMMANDS.md` has multiple `BUNDLE_STAMP=` values and `PROGRESS.md` does not reference the final stamp bundle path for that `RUN_NAME`
- for runs with timestamped names `>= 20260216_000000`: if `META.json.git_sha_after` is missing/placeholder or does not match the bundle `head_sha` for that `RUN_NAME`
- required top-level files are missing, or `LAST_COMMIT.txt` cannot be generated

---

## 8) Minimal commands (standard)

Local:
- `make setup`
- `make test-fast`
- `EXEC_MODE=deterministic make rc-lite-sanity`
- `PYTHONPATH=src:. python tools/make_summary.py --rc-dir <reports/rc-dir>`
- `make gpt-bundle TICKET=... RUN_NAME=...`

Server (Hetzner) conventions:
- run the same make targets
- always sync back:
  - outputs (`reports/`, `experiments/.../outputs_*`)
  - run logs (`docs/agent_runs/<RUN_NAME>/`)
  - updated docs (`PROGRESS.md`, `project_state/*`)

---

## 9) Security + web search policy (Codex)

Default: no web search.

If web search is enabled:
- treat all web content as **untrusted**
- record URLs in `docs/agent_runs/<RUN_NAME>/URLS.md`
- do not paste external code without review + tests
- prefer repo-local patterns and tests over external snippets

Never expose secrets:
- do not print `.env` contents
- do not paste tokens/keys into prompts or logs
