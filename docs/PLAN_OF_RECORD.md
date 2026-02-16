# PLAN OF RECORD — FJS De‑aliasing Overlay for Portfolio Risk (fjs-dealias-portfolio)

Last updated: 2026-02-16

**Ground-truth status references (must stay in sync):**
- `PROGRESS.md` (provenance for every run + change)
- `project_state/CURRENT_RESULTS.md` (latest validated drops)
- `project_state/KNOWN_ISSUES.md` (current blockers)
- `project_state/PIPELINE_FLOW.md`, `project_state/DATAFLOW.md`, `project_state/EXPERIMENTS.md`, `project_state/CONFIG_REFERENCE.md`
- Current external audit snapshot: `docs/gpt_outputs/20260216_analysis.md`
- Current external audit (full): `docs/gpt_outputs/20260216_analysis_full.md`
- Prompt-1 diagnosis context: `docs/gpt_outputs/20251222_prompt1_diagnosis.md`

---

## 0) Stop-the-line rules (non‑negotiables)

A run is **not mergeable** and **not citeable** (for advisor or paper) if any of these are violated:

- **No headlines from capped / truncated evals**:
  - Any run with `cap_active=true` is *non‑headline* and must be excluded from primary summary tables.
  - Capped runs are allowed only as smoke/debug and must be labeled everywhere (`cap_sources` + limitations).
- **No silent fallbacks, ever**:
  - Missing config files → **hard error**, not “defaults”.
  - Missing portfolio solver → **fail-loud** OR explicit **skip with reason** (never EW fallback).
  - Any “treatment collapses to baseline” must be logged with an explicit reason code.
- **No “treatment=never” results**:
  - If overlay acceptance ~0 or the overlay does not materially change Σ or weights, you cannot claim anything about its effect.
  - You must report acceptance and “changed-window” counts alongside any ΔLoss.
- **Every ticket must leave an audit trail**:
  - A run log under `docs/agent_runs/<RUN_NAME>/` (see `docs/DOCS_AND_LOGGING_SYSTEM.md`)
  - Tests run (`make test-fast` minimum) and recorded in the log + commit body.
- **Pipeline validity gates research**:
  - If validity fails, we stop and fix validity before running more grids.

### 0.1) 2026-02-16 recenter snapshot (priority override)

- The engineering pipeline is strong and auditable; this is not the blocker.
- Publishable research status is still blocked by weak real-data effect evidence.
- Injection sensitivity on real windows is still flat-zero in the latest week-design run.
- Immediate priorities are:
  1. Show non-flat injection response on at least one design, or conclusively explain the detection/residual mismatch.
  2. Ship one advisor-ready uncapped run with valid aligned comparisons and clear detection/acceptance reporting.
- Until those are done, do not expand experiment grids.

---

## 1) Crisp research framing (estimand, treatment, baselines, design)

### 1.1 Estimands (what we are actually evaluating)

For each rolling window \(t\), with returns \(R_t \in \mathbb{R}^{T \times p}\) (optionally factor‑prewhitened):

1) Estimate covariance \(\hat\Sigma_t\).
2) Produce portfolio weights \(w_t\) using a fixed portfolio rule.
3) Measure forward realized risk over horizon \(h\) (e.g., 21 trading days).

**Primary estimands (paper-grade):**
- Variance forecast losses (aligned windows only):
  - **MSE** of variance forecast error
  - **QLIKE** (variance forecast QLIKE)
- Portfolio operational metrics:
  - acceptance / detection rate
  - skip rate by reason
  - weight stability / turnover
  - fraction of windows where treatment changes Σ and/or \(w\)

**Optional (appendix / if implemented consistently):**
- tail metrics (VaR/ES) errors on forward returns

**Strict rule:** All Δ metrics and DM tests must use **the intersection of valid windows** (repo already enforces `comparison_valid_*` + `n_effective_*`).

### 1.2 Treatment vs baseline (what changes)

**Baseline:** a “base” covariance estimator \(\hat\Sigma^{base}_t\), e.g. shrinkage / spectral shrinkage / factor / robust scatter.

**Treatment:** **FJS/MANOVA de‑aliasing overlay**, applied **on top of** \(\hat\Sigma^{base}_t\), gated by detection diagnostics and guardrails.

- Detection produces a set of spike directions \(\{\hat v_i\}_{i=1}^k\) and aliased spike eigenvalues \(\{\hat\lambda_i\}\).
- De‑aliasing maps \(\hat\lambda_i \mapsto \hat\mu_i\) (e.g., via \(\hat\mu=\hat\lambda/t_r(\hat\lambda,a)\) as in your one‑pager).
- Overlay produces a corrected covariance:

\[
\hat\Sigma^{treat}_t
= \hat\Sigma^{base}_t \;+\; \sum_{i=1}^k \Delta_i \hat v_i \hat v_i^\top
\]

where \(\Delta_i\) is the implemented correction (must be documented exactly).

**Implementation anchors (must match the math above):**
- Detection/gating: `src/fjs/gating.py`, `src/fjs/mp.py`
- Overlay operator: `src/fjs/overlay.py` (**paper must write down the exact operator used here**)
- Robust edge modes: `src/fjs/robust.py`
- Base estimators: `src/baselines/covariance.py` (LW/OAS/RIE/QuEST etc), `src/finance/factors.py` (prewhitening + factor utilities)
- Portfolio solve: `src/finance/portfolios.py`

**Hard requirement:** If gating rejects or guardrails fire, treatment must no-op to baseline **with an explicit reason code**, not silence.

### 1.3 Minimal baseline set (publishable minimum)

We do not get publishability comparing “overlay vs SCM only”.

Minimum baselines for a defensible study:
- Shrinkage: **OAS**, **Ledoit–Wolf** (`src/baselines/covariance.py`)
- Spectral shrinker: **RIE/QuEST** (`src/baselines/covariance.py`)
- Factor baseline and/or factor prewhitening ablation:
  - prewhitening toggle (`src/finance/factors.py`) is mandatory as an ablation
  - if factor covariance estimator exists as a direct baseline, include it

### 1.4 Data design (what structure we assume)

Repo currently supports:
- **Daily eval runner**: `experiments/eval/run.py`
  - `group_design ∈ {dow, week, vol, dowxvol}`
- **Weekly equity panel**: `experiments/equity_panel/run.py`
  - `design ∈ {oneway, dow, vol, nested}`

**Plan-of-record position (blunt):**
- **Primary paper design should be `week` (daily) or `oneway` (weekly)** — many groups with small replicates is closer to the random‑effects “balanced design” intuition.
- **Daily DoW (`dow`) is secondary/ablation** because it’s only 5 groups; theory match is questionable and likely explains tiny empirical effects.
- **Nested weekly is “paper‑optional” until calibration coverage matches real p,T and real windows stop skipping.**

---

## 2) Minimal assumptions needed for FJS/MANOVA theory to be relevant

You do not need perfect assumptions; you need:
- High‑dimensional regime: \(p\) comparable to effective sample (so MP edge/outlier separation is meaningful).
- Balanced design (critical): equal replicates per group (one‑way) / per subgroup (nested).
- Random-effects / variance component structure is a usable approximation for the grouped returns (finance mapping is not literal; must be defended empirically).
- Calibration is fixed from **synthetic null/power** and not tuned on the evaluation set.
- Time dependence / heteroskedasticity is acknowledged and stress‑tested (block bootstrap, crisis slicing, robustness).

**Repo alignment (current):**
- Balanced-window construction exists (`src/eval/balance.py`).
- Synthetic calibration suite exists (incl. nested kill-test) under `experiments/synthetic/`.
- Prewhitening and robust scatter are first-class toggles (`project_state/CONFIG_REFERENCE.md`).

**Known broken/blocked:**
- Nested real-data windows can be skipped due to missing calibrated \((p,T)\) grid points (`calibration_missing_p_T` for p≈188, T=70/80). This blocks nested claims.

---

## 3) Minimal publishable package (what we must deliver)

### 3.1 “Pipeline validity” deliverable (engineering contribution, mandatory)

A calibrated, well‑instrumented overlay pipeline with:
- no silent fallbacks (config, solver, skip policies)
- attributable gating/skip reasons (no `guard_other` blob, no opaque `diagnostic_failure`)
- synthetic null‑FPR controlled at a declared target operating point
- run metadata + reproducibility (dataset hashes + git SHA + resolved config)

### 3.2 Real-data “paper v1” deliverable (minimum viable grid)

**One primary design + one ablation only** (do not explode the grid before this is clean).

Primary: daily `week` design (`experiments/eval/run.py`, Make: `make rc-week`)

Ablation: daily `dow` design (`make rc-dow`)

Fixed evaluation settings (unless advisor overrides):
- Universe: top‑60 assets (or current stable default)
- Window/horizon: 126×21
- Prewhitening: OFF vs ON (FF5+MOM), paired
- Edge modes: `scm` vs `tyler` (paired)
- Baselines: `oas`, `lw`, `quest/rie` (minimum)
- Treatment: overlay on top of each chosen baseline (or at minimum, overlay on top of a single pinned base + a robustness base)

Portfolios:
- EW (sanity)
- Constrained min‑var (MV): must log constraint binding, skips, solver status

**Required reporting for paper v1 runs:**
- ΔMSE, ΔQLIKE, DM tests + `n_effective_*`
- acceptance/detection rate + skip reason histogram
- **conditional effects** on windows where treatment changes Σ/weights
- crisis-sliced (2020/2022) safety table (even if secondary)

### 3.3 Synthetic deliverable (paper appendix, mandatory for credibility)

- One‑way null + power curves:
  - target null FPR (e.g., 2%) and power vs spike strength
- Nested kill-test:
  - must pass null‑FPR target at relevant p,T grid cells *before* nested real-data claims
- Injection sensitivity on *real windows*:
  - demonstrate detection/acceptance responds monotonically to injected spike magnitude under realistic noise

---

## 4) Acceptance criteria: “the pipeline is valid”

A run is “valid for research conclusions” only if ALL are true:

### 4.1 Tests
- `make test-fast` passes.
- Any changed behavior has a unit/integration test.

### 4.2 Data integrity
- `python tools/verify_dataset.py ...` passes for every dataset used.
- Dataset hashes are recorded in run metadata.

### 4.3 Config integrity
- Paper/RC targets must load a **real config file** (no defaults fallback).
- Missing config path → **hard error** with a clear message.
- Run metadata records: config path + config hash + resolved config dump.

### 4.4 No silent fallbacks
- MV solve:
  - missing solver → fail-loud OR explicit skip with `skipped=true` and `skip_reason=missing_solver`
  - never “equal-weight fallback”
- Window selection:
  - any skip due to caps/coverage/balance/conditioning must be counted + attributed.

### 4.5 Cap discipline
- Headline tables must exclude `cap_active=true` runs.
- Capped runs must surface cap sources in `limitations.md`.

### 4.6 Diagnostics are attributable
- No opaque buckets:
  - `guard_other` should be absent, or unreachable by construction.
  - `diagnostic_failure` must include exception type + stage + minimal context.
- Daily + weekly diagnostics must include:
  - `skip_reason_primary` counts
  - guard tallies
  - acceptance/detection rates

### 4.7 Calibration validity
- Synthetic null‑FPR ≤ target for all claimed modes (design × edge_mode × relevant p,T bins).
- Calibration artifacts are versioned, with audit metadata (run_name, timestamp, git_sha, config_hash).

---

## 5) Roadmap (with commands + required artifacts)

### Horizon 1: 1–2 weeks (debug/validity + advisor-ready RC run)

**Goal:** close the two blocking validity gates before any new grid expansion.

1) Injection flat-zero debugging (TOP PRIORITY; ticket-18 still open)
- Current evidence to explain:
  - `reports/inject_spike/20251226_ticket24_week_full_fix/curve.csv` shows detection=0 and acceptance=0 for all tested `mu`.
  - `reports/inject_spike/20251226_ticket24_week_full_fix/gating_reasons.csv` is dominated by pre-gate `tvec_off_component` and no-root reasons.
- Required outcome:
  - either a non-flat `mu -> detection/acceptance` curve on at least one design/mode, or
  - a conclusive, artifact-backed explanation for why detection math mismatches financial residuals.
- Minimum commands:
  - `make test-fast`
  - `make inject-spike` (or explicit CLI runs with frozen config + seed)

2) One advisor-ready uncapped run (ticket-20 still open)
- Required run properties:
  - `cap_active=false`
  - valid aligned comparisons (`comparison_valid_* = 1`)
  - meaningful `n_effective_*` and non-empty summary tables.
- Minimum commands:
  - `make test-fast`
  - `EXEC_MODE=throughput make rc-week RC_WORKERS=$(nproc)`
  - `PYTHONPATH=src:. python tools/make_summary.py --rc-dir reports/<rc-week-dir>`
- Required artifacts:
  - `run.json`, `resolved_config.json`
  - `summary/summary_perf.csv`, `summary/summary_detection.csv`, `summary/completeness.json`, `summary/limitations.md`
  - skip-reason and acceptance diagnostics safe for advisor review

3) Keep previously completed enablers stable (already done; do not regress)
- Config integrity fail-loud behavior (ticket-16)
- Nested calibration coverage update (ticket-17)
- Conditional changed-window reporting (ticket-19)

### Horizon 2: 4–8 weeks (full experiment grid + robustness checks)

**Only start after Horizon 1 is clean.**

- Full daily grid (minimal but complete):
  - designs: `week` (primary), `dow` (secondary), optional `dowxvol` if balance stable
  - prewhitening: off/on
  - edge modes: `scm`/`tyler`
  - baselines: `oas`, `lw`, `quest/rie`, (+ factor estimator if available)
  - gate modes: strict vs soft, calibrated vs fixed (if both exist)
- Robustness:
  - regime slicing: calm vs crisis (explicit 2020/2022 windows)
  - block bootstrap CIs for Δ metrics
  - sensitivity sweep over key guardrails (delta_frac, stability_eta)
- Required artifacts:
  - a single consolidated grid table (ΔLoss + DM + acceptance/skip)
  - crisis-only safety table
  - diagnostic plots (acceptance vs regime, guard distributions)

### Horizon 3: Paper-level (submission readiness)

- Lock a paper configuration:
  - pinned YAML configs + pinned calibration JSONs
  - dataset hashes and exact date ranges frozen
  - deterministic paper make target
- Write down the estimator precisely:
  - mathematically define overlay operator (must match `src/fjs/overlay.py`)
  - add a “operator verification” unit test (toy diagonal case + known spike)
- Add “assumption violation” simulations:
  - time dependence, vol clustering, heavy tails, factor mis-specification
- Produce camera-ready outputs:
  - main tables: ΔQLIKE/ΔMSE (+ tail metrics if defensible), acceptance/skip, FPR/power
  - appendix: ROC-like diagnostics, gating distributions, sensitivity plots

---

## 6) Standardized outputs for every valid run

Every “valid” run directory must contain:
- `run.json` (cap_active, cap_sources, dataset ids/hashes, git sha, config path/hash)
- `resolved_config.json` (resolved final config)
- `skip_stats.csv` (counts/shares by reason)
- `metrics.csv` / `metrics_detail.csv` (+ DM tables if enabled)
- `summary/summary_perf.csv`, `summary/summary_detection.csv`
- `summary/completeness.json`
- `summary/limitations.md` (auto text listing caps/skips/known failure modes)

---

## 7) Decision gates / pivot triggers (explicit)

- If (after Horizon 1 fixes) conditional ΔQLIKE/ΔMSE on changed windows is ~0 and acceptance is non-trivial:
  - pivot framing to: **“aliasing signatures are rare or already handled by shrinkage; gating prevents harm; conditions for benefit are X.”**
- If acceptance remains near zero even after injection shows detectability:
  - pivot to **design mapping**: the grouping structure is not producing the intended variance-component signal.
- If nested cannot be made live at real p,T without losing FPR control:
  - nested stays “future work” with documented failure analysis; paper v1 excludes it.
