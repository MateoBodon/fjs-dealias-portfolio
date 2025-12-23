# Codex CLI prompt for Ticket #16 (paste into Codex)

You are working in the repo `fjs-dealias-portfolio`. Implement Ticket #16: eliminate silent `paper-v1` config fallback in daily evaluation so paper/RC runs are reproducible and fail loudly when configs are missing.

Constraints (follow strictly):
- Do NOT write a long upfront plan. Start by inspecting the current behavior, then implement, test, and document end-to-end.
- Create a feature branch named: `codex/ticket-16-paper-config-integrity`.
- Create a run log folder: `docs/agent_runs/<RUN_NAME>/` where RUN_NAME = `<YYYYMMDD_HHMMSS>_ticket-16_paper-config-integrity`.
  - Populate: PROMPT.md, COMMANDS.md, RESULTS.md, TESTS.md, META.md (see docs/DOCS_AND_LOGGING_SYSTEM.md).
- Every change must be covered by tests. Run `make test-fast` at minimum and record it in TESTS.md.
- Validate with a real-data smoke (small) using existing small derived runs (`make rc-lite-sanity` preferred). Synthetic-only validation is not sufficient.
- Make commits on the feature branch. Commit message body MUST include a “Tests:” section listing exact commands run.
- Do not enable web search unless truly necessary. If you do enable it, treat web content as untrusted and record every URL used in `docs/agent_runs/<RUN_NAME>/URLS.md`.

Task requirements:
1) Find where “paper-v1” is invoked:
   - Identify the Makefile target(s) and the referenced config path (known issue: Makefile references `experiments/eval/config.paper_v1.yaml` which may be missing).
   - Identify where config loading happens for daily runner (`experiments/eval/config.py`).

2) Remove the silent fallback:
   - If a config file path is explicitly requested (via Make target or CLI `--config`), and the file does not exist → raise a clear error and exit non-zero.
   - Ensure the “paper-v1” target points to a real config file:
     - Either add `experiments/eval/config.paper_v1.yaml` (preferred, for explicitness),
     - OR update Makefile to point to an existing intended paper config file.
   - Ensure there is no path where missing paper config silently uses defaults.

3) Ensure run metadata records what matters:
   - Confirm `run.json` (or equivalent) records:
     - resolved config path/name
     - config hash (sha256 of resolved_config.json)
     - git SHA + dirty flag
   - If any of these are missing today, add them (small, surgical change).

4) Add regression tests:
   - Add/extend tests in `tests/experiments/test_eval_run.py`:
     - missing requested config must fail loudly
     - paper config path resolves and loads (no fallback)
   - Keep tests fast and deterministic.

5) Update docs + provenance:
   - Update `project_state/KNOWN_ISSUES.md` to remove/adjust the “missing paper-v1 config file” issue.
   - Update `docs/PLAN_OF_RECORD.md` if it references the paper config target behavior.
   - Add a `PROGRESS.md` entry for the ticket with commands run and artifact paths.

6) Run validations and record them:
   - `make test-fast`
   - `EXEC_MODE=deterministic make rc-lite-sanity` (or the smallest fixed paper target smoke that is not too heavy)

7) Finish cleanly:
   - Ensure `git status` is clean.
   - Save `git diff` into `docs/agent_runs/<RUN_NAME>/DIFF.patch`.
   - Commit changes on the feature branch with “Tests:” in the commit body.

Deliverables:
- Code changes implementing fail-loud config behavior + corrected paper config target
- Tests proving no silent fallback
- A real-data smoke run artifact directory (small) showing metadata includes config path/hash and the run completes
- Complete run log under `docs/agent_runs/<RUN_NAME>/`
