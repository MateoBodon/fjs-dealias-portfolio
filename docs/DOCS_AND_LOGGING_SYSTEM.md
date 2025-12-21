# DOCS + LOGGING SYSTEM (ENFORCED)

**Last updated:** 2025-12-21  
**Rule:** If it isn’t logged, it didn’t happen. No run log ⇒ no merge. No tests recorded ⇒ no merge.

---

## 1) Directory layout (hard-coded conventions)

### 1.1 Prompts (human + agent)
- `docs/prompts/`
  - `ticket-XX_<slug>.md` — canonical prompts used for Codex/agents
  - `analysis_notes_<slug>.md` — scratch notes (optional)
  - **Rule:** every agent run must copy its exact prompt into the run log.

### 1.2 Agent run logs (mandatory)
- `docs/agent_runs/<RUN_NAME>/`
  - `PROMPT.md` — exact prompt text (verbatim)
  - `COMMANDS.md` — every command executed (copy/paste-able)
  - `RESULTS.md` — what changed + what the results were (include file paths)
  - `TESTS.md` — tests run + pass/fail + runtime
  - `META.md` — provenance + config hashes + dataset ids + environment notes
  - optional:
    - `DIFF.patch` — `git diff` (recommended)
    - `NOTES.md` — debugging notes
    - `FAILURES.md` — stack traces / known failures

### 1.3 Bundles (advisor share / audit)
- `docs/gpt_bundles/`
  - produced via `make gpt-bundle TICKET=<ticket> RUN_NAME=<run>`
  - bundles must include: run log + diff + key docs (PLAN_OF_RECORD, etc.)

### 1.4 Output roots (do not mix)
- Daily eval RC outputs: `reports/<run_id>/`
- Weekly panel outputs: `experiments/equity_panel/outputs_*/.../`
- Synthetic outputs: `reports/synthetic/<run_id>/`

---

## 2) Run naming scheme (required)

### 2.1 RUN_NAME format
`YYYYMMDD_HHMMSS_ticket-XX_<slug>`

Examples:
- `20251221_031500_ticket-01_overlay-forensics`
- `20251221_104200_ticket-03_nested-calibration-grid`

### 2.2 Slug rules
- lowercase, hyphen-separated
- ≤ 6 words
- must reflect the primary change

---

## 3) What MUST be recorded (minimum metadata)

### 3.1 Code provenance
Record in `META.md`:
- git branch name
- git SHA (short + full if available)
- whether working tree was clean at start (`git status`)
- code signature hash if available (see `src/meta/run_meta.py:code_signature`)

### 3.2 Data provenance
Record:
- dataset id(s) and hash(es)
  - `data/registry.json` entry key for returns dataset
  - `data/factors/registry.json` entry key for factors dataset
- local data mirror / mount path (if using external storage) and how it is linked into `data/` (symlink or bind mount)
- external WRDS lake inventory (when available) so future refreshes can be traced
- verification command used:
  - `python tools/verify_dataset.py ...` (or the Make target that runs it)

#### External WRDS lake inventory (user-provided mirror)
If you have a full WRDS mirror available locally, record the root path + structure here.
This is **not** required for daily runs (we only need `data/returns_daily.csv` and
`data/factors/ff5mom_daily.csv`), but it is the source of truth for refreshes.

**Local mirror (macOS)**: `/Volumes/Storage/Data` (user-reported; not visible in CI)

```
/Volumes/Storage/Data
  wrds/
    raw/           # primary raw extracts
    manifests/     # ingestion manifests (timestamped)
    derived/       # (empty placeholder)
    meta/          # (empty placeholder)
    universes/     # (empty placeholder)

    raw/crsp/
      dsf/         # yearly partitions 2005–2024
      dsf_v2/      # yearly partitions 2005–2024
      dsenames.parquet
      dsedelist.parquet
      dsp500list.parquet
      dsp500list_v2.parquet
      msp500list.parquet
      ccmxpf_linktable.parquet
      ccmxpf_lnkhist.parquet

    raw/comp/
      funda/       # yearly partitions 2005–2025
      fundq/       # yearly partitions 2005–2025
      company.parquet

    raw/optionm/
      secnmd.parquet
      opprcd/      # yearly partitions 2005–2025
      secprd/      # yearly partitions 2005–2025
      distrd/      # yearly partitions 2005–2025
      zerocd/      # yearly partitions 2005–2025
      zero_curve/  # empty placeholder
      wrdsapps_opcrsphist.parquet  # OptionMetrics↔CRSP link

    raw/ff/
      factors_daily.parquet

    manifests/
      20251220_204920
      20251220_212339
      20251220_214010
      20251220_214218
      20251220_214458
      20251220_214625
      20251220_221935
      20251220_223429
      20251221_001039
      20251221_001618
```

**Usage note:** keep the mirror outside git; symlink into `data/` only when
regenerating `data/returns_daily.csv` or factor files.

### 3.3 Config provenance
Record:
- exact CLI invocation (including flags)
- resolved config file path (`resolved_config.json` or `config_resolved.yaml`)
- config hash (sha256 of resolved config file)
- random seeds used (if applicable)

### 3.4 Validity flags + metrics
For daily RC runs, record:
- `cap_active` and `cap_sources` from `run.json`
- `n_effective_mse`, `n_effective_qlike` and `comparison_valid_*`
- detection/acceptance rates
- top skip reasons (from `full/skip_stats.csv`)

For weekly runs, record:
- detection rate
- skip reason counts (`weekly_diagnostics.md`)
- `guard_unknown` count must be 0

### 3.5 Failures
- If any run fails: record the traceback (stage + exception type + message) and the exact command that caused it.

---

## 4) Summary docs that MUST be updated (when applicable)

### 4.1 PROGRESS.md (always if code changed)
Update with:
- timestamp
- branch/run name + git SHA
- commands executed
- tests run
- key results (one paragraph max)
- artifact paths

### 4.2 project_state/*
Update when results change materially or a blocker is resolved:
- `project_state/CURRENT_RESULTS.md`
- `project_state/KNOWN_ISSUES.md`
- `project_state/ROADMAP.md`
- `project_state/RESEARCH_NOTES.md` (if instrumentation/diagnostics changed)

### 4.3 docs/PLAN_OF_RECORD.md
Update when:
- we change the minimal publishable grid
- we make a pivot decision (e.g., drop nested from v1)
- we change acceptance criteria or “kill criteria”

### 4.4 CHANGELOG.md
Update only for user-facing behavior changes:
- new CLI flag semantics
- new required artifacts (e.g., `overlay_forensics.csv`)
- breaking changes to configs

---

## 5) Run log template (copy/paste)

Create these files at run start:

### PROMPT.md
- exact agent prompt (verbatim)

### COMMANDS.md
- list commands in the exact order executed
- include environment activation lines (venv, conda, etc.)

### TESTS.md
- `make test-fast` output summary
- any targeted pytest runs

### RESULTS.md
- bullets:
  - what files changed (paths)
  - what new artifacts exist (paths)
  - what the key metrics were (numbers + where in CSV they live)

### META.md
- git branch + SHA
- dataset ids + hashes
- config path + hash
- runtime environment notes (OS, Python version, deterministic mode, worker count)

---

## 6) Enforcement hooks (where the repo already supports this)

- `src/meta/run_meta.py` writes `run_meta.json` (use it; don’t reinvent).
- `src/meta/completeness.py` can assess daily/weekly run completeness; summaries must write `summary/completeness.json`.
- `tools/make_summary.py` is the choke point for “headline tables” — it must:
  - exclude capped/incomplete runs
  - emit `limitations.md`
  - surface `n_effective_*` and skip shares

---

## 7) Stop-the-line rules (non-negotiable)

You must NOT merge if any are true:
- no run log directory for the work you did
- tests not run (or not recorded in commit body)
- new outputs are generated but not referenced in RESULTS.md
- headline tables include `cap_active=true` runs
- any solver fallback occurs without explicit skip reason
- `guard_unknown > 0` in weekly diagnostics (means logging is incomplete)
