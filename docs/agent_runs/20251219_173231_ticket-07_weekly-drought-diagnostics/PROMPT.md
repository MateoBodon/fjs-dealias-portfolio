# Prompt

# AGENTS.md instructions for /root/fjs-dealias-portfolio

<INSTRUCTIONS>
# AGENTS.md

## Project summary

This repo implements an FJS-style MANOVA de-aliasing overlay on high-dimensional covariance estimates for equity return panels.

Key components:

- **Detection & overlay**  
  FJS-inspired spike detection and spectral transform overlay (`src/fjs/`) applied on top of base covariance estimators (SCM, shrinkage, robust SCM, factor-based) in `src/finance/`.

- **Synthetic harness**  
  Null/power calibration for group designs and edge modes in `experiments/synthetic/`, writing calibration artifacts into `calibration/` and `reports/synthetic/`.

- **Equity-panel runners**  
  Weekly group designs (DoW, nested, volatility-state) on WRDS-style return panels in `experiments/equity_panel/`, with configs under `experiments/equity_panel/config.*.yaml`.

- **Evaluation harness**  
  Daily portfolio risk evaluation (EW and MV portfolios; variance, VaR, ES, DM tests) in `experiments/eval/` + `src/evaluation/`.

- **Reporting**  
  RC galleries, memos, and briefs in `figures/rc/YYYYMMDD/` and `reports/rc-YYYYMMDD/`, assembled via `tools/build_*`.

You should treat `docs/LONG_TERM_PLAN.md` and `PROJECT_STATE/*.md` as the source of truth for vision, designs, and experiment grid.

---

## Setup commands

Always assume you are starting at the repo root.

### 1. Python environment

Use a virtualenv or conda env; Python 3.10+ is preferred.

```bash
# Create and activate a virtualenv (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# or on Windows (PowerShell)
python -m venv .venv
.\venv\Scripts\Activate.ps1
Install the project in editable mode with dev extras:
bash
Copy code
pip install --upgrade pip
pip install -e .[dev]
If the repo uses uv or poetry, prefer whatever README.md / CONFIG_REFERENCE.md tells you. If in doubt, default to pip install -e .[dev].
2. Tests
Before and after non-trivial changes, run:
bash
Copy code
# Fast unit tests (no heavy I/O)
make test-fast

# Full suite (can be slow; includes integration tests)
make test
# or, if you need granularity:
make test-integration
make test-slow
If make is not available, inspect Makefile and reproduce the underlying pytest commands. Respect markers (e.g., -m "not slow" for fast runs).
3. Synthetic harness (local or Hetzner)
To run a small synthetic calibration:
bash
Copy code
# Small null/power sweep for acceptance thresholds
make sweep:acceptance

# Rebuild thresholds into JSON artifacts
make calibrate-thresholds
For heavy calibration (multi-thousand trials), use Hetzner and an appropriate profile; see docs/HPC.md.
4. Small real-data run (WRDS, Hetzner)
Assumptions:
WRDS daily returns and factor CSVs are available under a mounted path (e.g. /mnt/wrds) and symlinked/linked under data/.
CONFIG_REFERENCE.md describes which config files map to which datasets.
A typical small sanity run:
bash
Copy code
# From repo root, on Hetzner
export EXEC_MODE=throughput  # if supported by Makefile
make rc-lite-sanity
This should:
Load a small universe and time span (e.g. config.smoke.yaml / config.rc-lite.yaml or equivalent).
Run DoW + vol-state designs.
Produce outputs under experiments/equity_panel/outputs_* and reports/rc-YYYYMMDD/.
If make rc-lite-sanity does not exist yet, consult Makefile and experiments/equity_panel/config.*.yaml. A minimal manual invocation looks like:
bash
Copy code
python -m experiments.equity_panel.run \
  --config experiments/equity_panel/config.smoke.yaml
Followed by:
bash
Copy code
python -m experiments.eval.run \
  --config experiments/eval/config.smoke.yaml
Always prefer existing Make targets when available.
Important files and directories
Top-level
pyproject.toml / setup.cfg / requirements*.txt — dependencies.
Makefile — canonical commands for tests, synthetic runs, and RCs.
CHANGELOG.md — summary of structural changes.
docs/LONG_TERM_PLAN.md — long-term research plan and experiment grid.
PROJECT_STATE/ (or docs/PROJECT_STATE/) — pipeline, dataflow, experiment status, etc.
docs/HPC.md — Hetzner / HPC instructions.
docs/AGENT_RUNS/ — per-sprint logs; you should append here.
Source code (src/)
src/fjs/ — de-aliasing overlay, MP edge estimation, acceptance logic.
src/finance/ — covariance estimators (SCM, shrinkage, robust SCM, factor-based).
src/evaluation/ — rolling metrics, DM tests, VaR/ES, regime splits.
src/report/ — table and figure assembly.
src/meta/ — utilities, caching, registry helpers.
Experiments
experiments/synthetic/ — null/power harness and calibration.
experiments/equity_panel/
run.py — weekly design runner for WRDS equity panels.
config.smoke.yaml, config.rc-lite.yaml, config.rc.yaml, config.crisis.*.yaml — configs for small, rc-lite, full, and crisis runs.
outputs_* — output directories per run.
experiments/eval/
run.py — daily evaluation runner for EW/MV, VaR/ES.
config.*.yaml — evaluation configs.
Data
data/
returns_daily.csv or similar — daily equity return panel.
factors/*.csv — FF5+MOM or other factor panels.
registry.json — dataset digests (if present).
wrds/ — raw WRDS exports (should be in .gitignore).
Reports & figures
reports/rc-YYYYMMDD/ — each RC has:
summary.json, metrics_summary.csv, detection_summary.csv, etc.
memo.md, brief.md.
figures/rc/YYYYMMDD/ — figures for that RC.
reports/synthetic/ — null/power and calibration artifacts.
Tests
tests/ — unit and integration tests; markers distinguish fast/slow/integration.
Code style and conventions
Language & formatting
Python 3.10+.
Prefer black/ruff-compatible style if config exists.
Type hints for public functions where reasonable.
Keep functions short and composable; avoid mega-scripts.
Imports & structure
Use absolute imports (src.-style) rather than deep relative imports when possible.
Keep experiment scripts thin; core logic should live under src/.
Config-driven experiments
Do not hard-code experiment-specific parameters inside src/ modules.
Instead, rely on YAML config files in experiments/**/config.*.yaml.
Always log the effective config into output directories (config_resolved.yaml or equivalent).
Naming
Use meaningful names reflecting the theory:
edge_margin, q_max, delta_frac_min, etc. for gating.
design / grouping for DoW, nested, vol-state.
Prefer snake_case for variables and functions.
Testing and validation
General rules
After any non-trivial code change, run:
bash
Copy code
make test-fast
If you modify experiment/eval code:
make rc-lite-sanity (or the smallest relevant real-data config).
For changes affecting detection/overlay, synthetic harness, or calibration:
bash
Copy code
make sweep:acceptance
Use a reduced trial count if necessary for speed; document this in docs/AGENT_RUNS/*.
Real-data validation
Prefer tests that touch real WRDS-based data when feasible:
At minimum, make rc-lite-sanity on a small subset.
Check that:
Outputs are generated (metrics_summary.csv, detection_summary.csv).
Gating and detection metrics look sane (no obvious zero-coverage or 100% substitution due solely to caps).
Failure handling
If tests fail:
Capture the failing command and traceback in the corresponding docs/AGENT_RUNS/*.md section.
Do not auto-fix by weakening tests unless explicitly instructed; instead, propose changes in your notes.
Data and safety rules
You must treat data and infrastructure as fragile and non-reprovisionable.
WRDS data and credentials
Never commit WRDS raw data or any proprietary data to git.
Never commit secrets (WRDS, SSH keys, tokens, etc.) or config files containing credentials.
Assume data/wrds/ or /mnt/wrds is a mount or symlink and must not be modified.
Data directories
Do not delete or overwrite anything under data/ except:
Explicitly documented temp/cache paths (e.g., data/cache/, data/tmp/).
If a cleaning script exists (e.g., tools/clean_outputs.py), inspect it before running and prefer targeted cleaning.
Outputs
Do not rm -rf experiments/**/outputs_*, reports/, or figures/ without explicit instruction.
When you must regenerate outputs:
Write into new timestamped directories (outputs_YYYYMMDD_HHMMSS).
Do not overwrite previous RCs.
Remote execution (Hetzner)
Assume the remote filesystem contains long-running calibration and RC artifacts; do not mass-delete.
Do not change system-level packages or OS configuration.
Agent workflow and house rules
1. Models and profiles
Default model: GPT-5.1-Codex-Max.
When running on Hetzner via Codex CLI:
Use the fjs-hetzner profile if available (see ~/.codex/config.toml and docs/HPC.md).
That profile is allowed to run heavy jobs but still must respect data rules.
2. Change process
For each substantial sprint or series of edits:
Read context
Read:
docs/LONG_TERM_PLAN.md
AGENTS.md (this file)
Relevant PROJECT_STATE/*.md files for the area you’re modifying.
Create or update sprint log
Use docs/AGENT_RUNS/<date>_codex_sprint_<N>.md:
If a sprint file for today already exists, append to it.
Otherwise, create a new one with:
Goals.
Planned tasks.
Commands you expect to run.
Plan 3–7 outcome-level tasks
Examples:
“Implement rc-lite-sanity and verify it runs on Hetzner.”
“Increase nested design coverage; confirm non-zero accepted detections.”
“Add test coverage for factor-based prewhitening.”
Write these into the sprint log before making changes.
Make small, reviewable changes
Use apply_patch (or equivalent) to edit files in small increments.
Prefer many small commits over one giant one.
Keep changes logically grouped (e.g., tests + code + docs for one feature).
Run tests
At minimum:
make test-fast
If you modified experiment/eval code:
make rc-lite-sanity (or the smallest relevant real-data config).
If you changed public APIs, config schemas, or experiment grids:
Update CHANGELOG.md.
Update relevant PROJECT_STATE/*.md (e.g., EXPERIMENTS.md, CURRENT_RESULTS.md, ROADMAP.md).
Branching and commits
Work on a dedicated branch, typically codex/<short-task>, e.g.:
codex/nested-diagnostics
codex/rc-lite-sanity
Use clear commit messages, e.g.:
feat: add rc-lite-sanity target
fix: relax nested guardrails for sparse years
docs: update LONG_TERM_PLAN and HPC instructions
3. When to stop and ask for human input
If you encounter:
Missing WRDS data or broken symlinks.
Ambiguous configs (e.g., conflicting design names).
Major performance regressions or unexplained test failures.
Then:
Stop making further changes.
Document:
The issue.
What you tried.
Reproduction commands.
Write a clear “next questions for the human” section at the end of the sprint log.
4. Non-destructive by default
Never:
Delete data directories not explicitly marked as safe to delete.
Force-push to main or master.
Weaken tests to “make things pass” silently.
Prefer:
Adding new targets/configs instead of overwriting old ones.
Guarded migrations (e.g., support old and new config keys with deprecation warnings).
If you are unsure whether an action is safe, assume it is not and write it down in docs/AGENT_RUNS/... for the human to decide.
</INSTRUCTIONS>

<environment_context>
  <cwd>/root/fjs-dealias-portfolio</cwd>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>bash</shell>
</environment_context>

You are Codex running in Codex CLI inside the repo workspace.

Ticket: ticket-07 (weekly detection drought diagnostics)

Non-negotiable constraints:
- AGENTS.md is binding. Read it first and follow stop-the-line rules.
- Do NOT change research semantics/thresholds in this ticket. This is diagnostics-only (opt-in) unless you discover an unambiguous bug; if you fix a bug, add a regression test and keep behavior changes narrowly scoped and documented.
- No silent fallbacks: if a required dependency/config/dataset is missing for the smoke run, fail loudly and record it.
- Heavy documentation: every command must be recorded; every result must point to artifacts; update project_state docs if results/validity statements change.

Work mode:
- Do NOT write a long upfront plan. Immediately explore the repo, implement the smallest correct change, run tests + smoke, and document outcomes.
- Make small logical commits on a feature branch. Each commit message body MUST include a “Tests:” line listing exact commands run.
- Finish by generating a review bundle via `make gpt-bundle TICKET=ticket-07 RUN_NAME=<RUN_NAME>` and record the bundle path in the run log.

Step 0 — Branch + run log (must happen first)
1) Create a feature branch: `codex/ticket-07-weekly-drought-diagnostics`.
2) Set RUN_NAME = `$(date +%Y%m%d_%H%M%S)_ticket-07_weekly-drought-diagnostics`.
3) Create `docs/agent_runs/<RUN_NAME>/` with:
   - PROMPT.md (paste this full prompt)
   - COMMANDS.md (append every shell command verbatim)
   - RESULTS.md (bullet results + artifact paths)
   - TESTS.md (tests executed + pass/fail)
   - META.json (ticket, run_name, created_at, git_sha)

Also update `docs/CODEX_SPRINT_TICKETS.md` to mark ticket-06 DONE (use info from the existing ticket-06 bundle) and add a new row for ticket-07 as “in-progress (run <RUN_NAME>)”.

Step 1 — Explore (fast, parallel)
- Use `rg` to find:
  - where weekly (equity_panel) runs compute detections and write manifests/metrics
  - where gating decisions + skip reasons are produced (likely in src/fjs/gating.py or nearby)
  - existing “skip_reason” plumbing (if any)
- Identify the smallest existing weekly smoke config (look in `experiments/equity_panel/` and/or `docs/agent_runs/*/config*.yaml`).
- Identify how output directories are structured and where to place diagnostics artifacts so they’re captured in manifests.

Step 2 — Implement diagnostics artifact (opt-in)
Implement an opt-in “weekly gating diagnostics” output:
- Add a config/flag (prefer config key, e.g., `diagnostics.gating_trace: true`) that when enabled causes each evaluated window to emit a structured diagnostics record.
- Diagnostics record requirements (per window):
  - window identifier (date range / index), p, T, estimator id, design id (DoW vs nested)
  - MP edge (or robust edge) value used, and summary of top eigenvalues (at least λ1..λ5 or λ1 and λ1/edge)
  - candidate count and best-candidate stats used by gate
  - each guardrail statistic that can reject (e.g., isolation score, eta/angle, off-leak, eps, delta_frac threshold)
  - final decision accepted/rejected + one canonical skip_reason string
- Write to a single artifact in the run output dir:
  - `gating_diagnostics.csv` (preferred) or `gating_diagnostics.jsonl`.
- Add a small helper to aggregate + summarize:
  - Create `tools/summarize_weekly_diagnostics.py` that reads the artifact and writes `weekly_diagnostics.md` (top skip reasons + counts, detection rate, and min/median/max of key gate stats).
  - Keep dependencies minimal (stdlib + pandas if already in repo).

Step 3 — Tests (minimum + new)
- Add a unit/integration test that:
  - Runs the smallest possible evaluation path (can be synthetic fixture) with diagnostics enabled
  - Asserts the diagnostics artifact exists and contains required fields/columns
  - Asserts skip_reason is never empty when accepted==false
- Run: `source .venv/bin/activate && make test-fast`
- Record all tests in `docs/agent_runs/<RUN_NAME>/TESTS.md` and in commit bodies.

Step 4 — Deterministic smoke runs (must do both)
A) Synthetic (schema exercise):
- Run a tiny synthetic job that yields at least one candidate window and produces diagnostics.
- Keep it very small (few windows/trials) to avoid long runtimes.

B) Real-data weekly smoke (the actual target):
- Run the smallest existing `experiments/equity_panel` weekly config with diagnostics enabled.
- Use deterministic execution mode (respect repo conventions; e.g., EXEC_MODE=deterministic if used elsewhere).
- Even if detections remain 0, diagnostics MUST be produced and must clearly show which gating constraint binds.

After smoke runs:
- Run `tools/summarize_weekly_diagnostics.py` on the real-data output dir and save the markdown summary alongside the run outputs (and/or in `reports/weekly_diagnostics_<RUN_NAME>/`).

Step 5 — Update state docs (only after artifacts exist)
- Update `project_state/CURRENT_RESULTS.md`:
  - Add a bullet summarizing what the diagnostics show (top 3 skip reasons + one sentence interpretation).
  - Reference the output directory and RUN_NAME.
- Update `project_state/KNOWN_ISSUES.md`:
  - Replace vague “weekly drought” item with a concrete diagnosis from the diagnostics artifact (or explicitly say “still unknown, diagnostics added” if the smoke run fails).
- Update `PROGRESS.md` with a dated entry including: goal, key findings, run name, and bundle path.

Step 6 — Bundle + finish
- Ensure `git status` is clean except intended changes.
- Generate bundle:
  - `make gpt-bundle TICKET=ticket-07 RUN_NAME=<RUN_NAME>`
  - Save `unzip -l <bundle.zip> > docs/agent_runs/<RUN_NAME>/bundle_contents.txt`
- Record bundle absolute path in `docs/agent_runs/<RUN_NAME>/RESULTS.md`.

Stop conditions (ask human / stop-the-line):
- If you discover diagnostics requires changing core gating behavior (thresholds, acceptance logic), STOP and ask before proceeding.
- If the only way to make diagnostics work is to enable network access or YOLO mode, STOP.
- If the weekly smoke depends on missing proprietary data, STOP and switch to the repo’s smallest derived dataset path; if none exists, document that as a blocker and add a TODO + failing test placeholder (but do not fake results).
