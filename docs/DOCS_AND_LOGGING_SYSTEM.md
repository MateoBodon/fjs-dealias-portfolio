# DOCS_AND_LOGGING_SYSTEM
Date: 2025-12-17  
Purpose: enforce **traceability** (every claim ↔ a run directory ↔ a config hash ↔ a git SHA ↔ tests).

This protocol applies to:
- human runs (Hetzner / local)
- Codex agent runs (CLI or IDE)

---

## 1) Directory conventions (source of truth)
### 1.1 Prompts (what we asked the agent/model to do)
- Location: `docs/prompts/`
- Naming:
  - `YYYYMMDD_HHMMSS_ticket-##_slug.md`
  - Example: `20251218_093000_ticket-01_nested-null-fpr.md`
- Contents MUST include:
  - ticket ID + goal
  - constraints / stop-the-line rules relevant to task
  - required tests/commands to run
  - required artifacts to produce
  - “definition of done” checklist

### 1.2 Agent runs (what actually happened)
- Location: `docs/agent_runs/<RUN_NAME>/`
- Run name format:
  - `YYYYMMDD_HHMMSS_ticket-##_slug`
  - Example: `20251218_093000_ticket-01_nested-null-fpr`
- Required files inside each run dir:
  - `PROMPT.md` (exact prompt used; copy from `docs/prompts/…`)
  - `COMMANDS.md` (commands executed; include working dir + env vars)
  - `RESULTS.md` (short summary + links to artifacts)
  - `DIFF.patch` (optional but strongly preferred; `git diff > DIFF.patch`)
  - `TESTS.md` (exact tests run + outcomes; copy into commit body too)
  - `META.json` with:
    - `git_sha` (HEAD)
    - `branch`
    - `config_hashes` (sha256 of relevant resolved configs)
    - `dataset_ids` (sha256 from registries or computed)
    - `exec_mode` (deterministic/throughput)
    - `host` (hetzner/local)
    - `start_time`, `end_time`
    - `failures` (list; empty if none)

### 1.3 Experiment run outputs (existing repo convention)
Do not fight the repo’s output structure; reference it.
- Daily eval outputs: `reports/<run_id>/...` (from `experiments/eval/run.py`)
- Weekly equity panel outputs: `experiments/equity_panel/outputs_*/...`
- RC-style consolidated drops: `reports/rc-<DATE>/` and `reports/rc-<DATE>-sanity-<STAMP>/`
- Calibration: `reports/synthetic/…`, `reports/figures/…`, plus committed JSONs under `calibration/` and `calibration_defaults.json`

**Rule:** every PR that changes behavior must reference at least one concrete output directory in `docs/agent_runs/.../RESULTS.md`.

---

## 2) Run naming and IDs (no ambiguity)
### 2.1 Canonical run ID for experiments
- Prefer existing runner-generated IDs/stamps (e.g., rc-lite-sanity stamp) and record them verbatim in `META.json` and `RESULTS.md`.
- When writing new scripts (synthetic kill-tests, etc.), require a `--run-id` or timestamped out dir and write a `run.json` containing `git_sha`, config hash, and seed.
  - Pattern already used in `experiments/synthetic/calibrate_thresholds.py` (see README + `project_state/CONFIG_REFERENCE.md`).

### 2.2 Config hash definition
- For any run directory that writes a `resolved_config.json`, define:
  - `config_hash = sha256(resolved_config.json bytes)`
- If a run uses multiple configs (daily + weekly), store all hashes in `META.json`.

---

## 3) What MUST be recorded (minimum viable audit trail)
For any claim that goes into `REPORT.md`, `METHODS.md`, a memo, or advisor communication, we must have:

### 3.1 Code identity
- `git_sha` and branch name
- whether there are local uncommitted changes (must be “clean” for final runs)

### 3.2 Data identity
- dataset sha256 from registries:
  - `data/registry.json`
  - `data/factors/registry.json`
- if a dataset is not in registry: compute sha256 and add it (or stop the run)

### 3.3 Execution identity
- `EXEC_MODE` and thread caps (BLAS/OpenMP)
- worker count, cache dir, resume flags
- solver backend used for MV (must be recorded)

### 3.4 Results identity
- paths to output dirs
- key metrics (ΔMSE EW/MV, detection/acceptance, percent_changed, turnover)
- failures/warnings (missing deps, skipped windows, PSD clipping, caps)

---

## 4) Summary docs we update (and when)
### 4.1 Always update on behavior changes
- `PROGRESS.md` (human-readable running log)
  - Add an entry: date, branch/sha, what changed, which run IDs validate it.
- `project_state/KNOWN_ISSUES.md`
  - Move issues from “known” → “fixed” only if validated by a run log + tests.
- `project_state/CURRENT_RESULTS.md`
  - Only update with runs that satisfy validity criteria (no partial runs, no silent fallbacks).

### 4.2 Optional but recommended
- `CHANGELOG.md` (repo root or `project_state/CHANGELOG.md`)
  - Summarize user-facing behavior changes with links to run logs.
- `docs/PLAN_OF_RECORD.md`
  - Only if roadmap/acceptance criteria change.

---

## 5) Stop-the-line logging rules (enforced)
A PR is **not mergeable** unless:
- `docs/agent_runs/<RUN_NAME>/` exists for the change (even if minimal)
- commit body includes:
  - `Tests: ...` (exact commands)
  - `Artifacts: ...` (paths to output dirs)
- any caps/fallbacks are either removed or explicitly recorded + defaulted off

---

## 6) Templates
### 6.1 `docs/agent_runs/<RUN_NAME>/RESULTS.md` template
- Goal:
- Summary outcome:
- Artifacts:
  - `<path1>`
  - `<path2>`
- Key metrics:
  - ΔMSE(EW):
  - ΔMSE(MV):
  - detection/acceptance:
  - percent_changed:
- Failures / warnings:
- Follow-ups:

### 6.2 Commit body template
- Tests: `make test-fast` ; `pytest -m integration -k ...`
- Artifacts: `reports/...` ; `experiments/...`
- Notes: (brief; include any behavior changes)

---
