# PLAN OF RECORD (FJS De-aliasing × Portfolio Risk)
Date: 2025-12-17  
Scope: This document is the **authoritative plan** for turning this repo into a publishable study (or deciding to pivot fast).  
Sources for current state: `project_state/CURRENT_RESULTS.md`, `project_state/KNOWN_ISSUES.md`, and the diagnosis in `fjsprompt1output.md`.

---

## 0) One-line framing (what this paper is)
We test whether **selective MANOVA / FJS “de-aliasing”** (a spike-detection + covariance substitution overlay) improves **out-of-sample portfolio risk forecasts** versus strong covariance baselines, under **high-dimensional, short-window** equity data with **approximately balanced group designs** (DoW / vol-state; nested is exploratory until proven safe).

---

## 1) Ground truth snapshot (what is true today)
**As of 2025-12-17:**
- Daily `rc-lite-sanity` overlay looks **harmful** on DoW and vol-state slices (ΔMSE>0; high “flip” rates).  
  Source: `project_state/CURRENT_RESULTS.md`.
- Weekly DoW + nested smoke runs show **0 detected/accepted windows** (“detection drought”).  
  Source: `project_state/CURRENT_RESULTS.md`, `project_state/KNOWN_ISSUES.md`.
- The **nested synthetic kill-test** shows **null acceptance ≈ 100% (FPR≈1.0)** → nested gating is currently **unsafe / invalid**.  
  Source: `project_state/CURRENT_RESULTS.md` (“Nested synthetic kill-test”), `project_state/KNOWN_ISSUES.md`.
- There are known validity hazards:
  - **Silent MV solver fallback** when `cvxpy` missing (falls back to EW with `converged=False`).  
    Source: `project_state/KNOWN_ISSUES.md`.
  - **Percent_changed ≈ 100%** is currently contaminated by capping behavior in RC-style runs (debug cap leaking into evaluation interpretation).  
    Source: `fjsprompt1output.md` (“evaluation contamination via caps”).

**Stop-the-line consequence:** until nested null FPR is fixed, nested must not be used for “main results”.

---

## 2) Research framing (estimand, treatment, baselines)

### 2.1 Estimand (what we evaluate)
Let `Σ_{t}` denote the (unknown) conditional covariance of asset returns over forecast horizon `H` (weekly or daily).  
We evaluate risk-forecast quality via:
- **Equal-weight (EW) portfolio** realized variance forecast error:
  - Forecast: `\hat v_t = w_EW^T \hat Σ_t w_EW`
  - Realized: `v_t = w_EW^T Σ_{t,H} w_EW` (proxied by realized returns over the horizon)
  - Metric: `MSE( \hat v_t, v_t )` and **ΔMSE** vs baseline.
- **Minimum-variance (MV) portfolio** realized variance (subject to constraints/regularization):
  - `\hat w_t = argmin_w w^T \hat Σ_t w + penalties, s.t. box/turnover/…`
  - Evaluate realized variance and forecast quality.
- Secondary:
  - QLIKE-style volatility loss (if implemented), VaR/ES coverage diagnostics, turnover, condition number, and stability metrics.
  - DM/sign tests **only if** sample sizes and assumptions are defensible; otherwise treat as descriptive.

### 2.2 Treatment (what “de-aliasing overlay” means in this repo)
**Treatment = selective overlay**:
- Compute a MANOVA/FJS spike estimate `\hat Σ^{FJS}_t` from a balanced-group design.
- Apply gating / guardrails → accept/reject spikes.
- If accepted, **substitute** (fully or partially) into a baseline covariance:
  - `\hat Σ^{treat}_t = Overlay( \hat Σ^{base}_t, \hat Σ^{FJS}_t; θ )`
- If rejected, use baseline: `\hat Σ^{treat}_t = \hat Σ^{base}_t`.

Code locus:
- Detection: `src/fjs/dealias.py`, `src/fjs/mp.py`, `src/fjs/balanced*.py`
- Gating: `src/fjs/gating.py`
- Overlay: `src/fjs/overlay.py`
- Runners: `experiments/equity_panel/run.py` (weekly), `experiments/eval/run.py` (daily)

### 2.3 Baselines (minimal set we will treat as “serious”)
We will not compare against a zoo. Minimal publishable set:
- **Shrinkage**: Ledoit–Wolf (`lw`), OAS (`oas`)  
- **Factor model**: FF5+MOM residual covariance (`factor`) + optionally a simple PCA factor estimator if already present
- **Robust/shrinkage hybrid**: Tyler-shrink (if already implemented: `tyler_shrink`)
- Optional “extra” baselines only if cheap & stable: `rie`, `poet`, `cc` (but `cc` must be clearly labeled and not used to hide instability)

Source for baseline menu: README (`make rc` baseline list) + `project_state/CONFIG_REFERENCE.md`.

---

## 3) Data design (what structure we assume)

### 3.1 Weekly equity panel (MANOVA design on balanced Week×Day cube)
Primary weekly design for theory alignment:
- **One-way / DoW**: groups `j ∈ {Mon,…,Fri}`; replicates are weeks within the rolling window.
- Balanced panel construction + partial-week policy: `experiments/equity_panel/run.py`, `project_state/PIPELINE_FLOW.md`.
- Typical config knobs: `window_weeks`, `horizon_weeks`, `stride_windows`, `assets_top`, `partial_week_policy`, `design {oneway,dow,vol,nested}`.

### 3.2 Daily evaluation (approximate balanced groupings)
Daily runner forms groups on the date axis:
- DoW grouping: `group_design=dow`
- Vol-state grouping: `group_design=vol` or `dow_vol` (based on EWMA volatility quantiles)
- This is *not* a clean i.i.d. MANOVA design; it’s an approximation used for empirical “does it help?” evaluation.

Code locus: `experiments/eval/run.py`, grouping in `experiments/daily/grouping.py` (see `project_state/PIPELINE_FLOW.md`).

### 3.3 Prewhitening / factors
Default mainline uses FF5+MOM prewhitening:
- Rationale: reduce common factor structure so MANOVA spikes are less likely to be “just the market”.
- Must log and check R², residual diagnostics.  
Source: `project_state/PIPELINE_FLOW.md`, README “prewhiten telemetry”.

---

## 4) Minimal assumptions required for theory to apply (and whether we meet them)

### 4.1 Assumptions (minimal set we need to state explicitly)
For MANOVA/MP-edge-based spike detection and de-aliasing to be theoretically meaningful:
1. **Balanced design**: group sizes/replicates are equal (or handled with correct weights).  
2. **Within-window stationarity**: covariance is approximately constant inside each rolling window.
3. **Noise structure**: residuals are approximately elliptically symmetric with weak dependence; MP edge approximations hold (or robustified).
4. **High-dimensional regime**: ratios like `p/T` are O(1), consistent with MP asymptotics.
5. **Calibration validity**: gating thresholds control null FPR at target (e.g., 2%) for the relevant `(p,T,design)` regimes.

### 4.2 Alignment status (repo reality)
- Balanced design: **partially met** (balancing + drop/impute exists), but must quantify deviation per window.  
  TODO: “design validity diagnostics” (see roadmap).
- Stationarity / dependence: **violated** in finance; mitigations: prewhitening + robust edges + regime splits.  
  Must be framed as “robust empirical evaluation”, not pure theory.
- Calibration: **met for one-way synthetic** (calibration defaults exist), but **NOT met for nested** (FPR≈1.0 under null).  
  Source: `project_state/CURRENT_RESULTS.md`, `project_state/KNOWN_ISSUES.md`.

**Therefore:** publishable story must treat nested as exploratory until fixed and validated.

---

## 5) Minimal publishable experiment set (what we will actually run)
If we cannot get signal here, we pivot.

### 5.1 Main experiments (must-have)
**Designs**
- Daily: DoW, vol-state (calm vs crisis splits) — `make rc-lite-sanity` is the canonical entry.
- Weekly: DoW (one-way) on at least one full-year slice + a crisis slice.

**Estimators**
- Baselines: `{lw, oas, factor, tyler_shrink (if stable)}`
- Treatment: `dealias` overlay on each baseline (or a single canonical baseline if we must simplify).

**Edge modes (robustness)**
- `edge_mode ∈ {scm, tyler}` (huber optional later)

**Metrics**
- ΔMSE for EW and MV risk forecast
- detection_rate, acceptance_rate, substitution_fraction
- percent_changed (must be computed without cap contamination)
- turnover and constraint binding stats for MV

### 5.2 Kill criteria (what counts as “working” vs “dead”)
We proceed only if:
- **Validity:** calibration + gating are not broken (no unconditional acceptance, no hidden fallbacks).
- **Signal:** at least one regime (calm or crisis) shows non-worse ΔMSE *or* a clear story like “overlay helps only in calm / only with prewhitening / only with robust edge”.
- If overlay is uniformly harmful across regimes **after** validity is fixed → pivot framing (see §9).

---

## 6) Pipeline validity acceptance criteria (binary pass/fail)
A run is **invalid** if any of these are violated.

### 6.1 Must-pass checks (for every RC / advisor run)
- `make test-fast` passes (or documented exception with failing test IDs).
- Run directory contains:
  - `resolved_config.json` (or equivalent)
  - `run_manifest.json` (git SHA, dataset ids, config hash, exec_mode, thread caps)
  - `metrics*.csv` / `summary.json`
  - `detection_summary.csv`
  - `diagnostics.csv` (+ detail if enabled)
- Run discovery/summary tooling flags incomplete runs as incomplete and excludes from aggregates.  
  (Known current failure: partial RC dirs not flagged; see `project_state/KNOWN_ISSUES.md`.)

### 6.2 Stop-the-line rules (hard)
- **No silent solver fallback:** if MV requested and solver missing/unavailable → fail the run loudly.  
  (Fix required; see `project_state/KNOWN_ISSUES.md`.)
- **No evaluation caps by default:** any `max_windows` / “cap to first K windows” must default OFF and be recorded when ON.  
  (Fix required; see `fjsprompt1output.md`.)
- **Nested design disabled for “main results” until:** nested synthetic null FPR is controlled (≤5% at target settings) and nested real-data coverage is nonzero.

### 6.3 Calibration sanity
- One-way synthetic null harness: realized FPR within tolerance of target (e.g., 2% ± 0.5% on >= 600 trials) for the `(p,T)` regimes used in RC.  
  Command anchor: README `make sweep:acceptance`.
- Nested synthetic kill-test (binary): null acceptance must not be near 1.0; must be <= 0.05 (or whatever target is selected) under the config used in nested runs.  
  Config anchor: `experiments/synthetic/config.nested.killtest.yaml`.

---

## 7) Roadmap: next 1–2 weeks (debug + validity + advisor-ready RC)
Goal: a “minimum credible RC” that an advisor can trust (even if the conclusion is “overlay is harmful”).

### 7.1 Ticketed work (must complete)
1) **Fix nested synthetic null FPR (FPR≈1.0)**  
   - Code: `src/fjs/gating.py`, `src/fjs/dealias.py`, `src/fjs/mp.py`, `experiments/synthetic/nested_killtest.py`  
   - Output: `reports/synthetic_nested_killtest/<RUN_ID>/summary.md` with null acceptance <= 5%  
   - Update: `project_state/KNOWN_ISSUES.md`, `project_state/CURRENT_RESULTS.md`

2) **Eliminate evaluation contamination via caps**  
   - Make caps explicitly “debug only”, default OFF; record in run metadata when used  
   - Code: `experiments/eval/config.py`, `experiments/eval/run.py`, `tools/make_summary.py`, `tools/summarize_rc_sanity.py`  
   - Acceptance: percent_changed no longer trivially 100% due to cap

3) **Fail-loud MV solver + record solver backend**  
   - Code: `src/finance/portfolio.py` (or solver wrapper), `experiments/eval/run.py`, `experiments/equity_panel/run.py`  
   - Acceptance: missing `cvxpy` causes explicit error if MV requested; metadata records solver used

4) **Add “detection drought” diagnostics for weekly/nested**  
   - Write per-window debug artifact: max eigenvalue, MP edge, candidate count, rejection reason  
   - Code: `src/fjs/dealias.py`, `src/fjs/gating.py`, `src/report/plots.py`, `tools/summarize_run.py`

5) **Make summary tooling completeness-aware**  
   - Flag partial RC dirs, avoid counting them; include vol-state sections consistently  
   - Code: `tools/summarize_rc_sanity.py`, `tools/make_summary.py`, `src/meta/run_meta.py`

### 7.2 Advisor-ready “RC run” definition (what we must produce)
Run: `make rc-lite-sanity` (deterministic) with caps OFF by default.  
Command anchor:
- `EXEC_MODE=deterministic make rc-lite-sanity`
Artifacts to present:
- `reports/rc-<date>-sanity-<stamp>/summary_sanity.json`
- `reports/rc-<...>/regime.csv`
- A memo/brief regenerated for the same stamp (no stale memo).

---

## 8) Roadmap: 4–8 weeks (full experiment grid + robustness)
Goal: enough controlled experiments to support a paper claim (positive or negative).

### 8.1 Experiment grid (minimal but sufficient)
- Daily: group_design ∈ {dow, vol, dow_vol}; prewhiten ∈ {off, ff5mom}; edge_mode ∈ {scm, tyler}; baseline ∈ {lw, oas, factor}.  
- Weekly: design=DoW one-way; window lengths ∈ {6, 13, 26} weeks; horizon ∈ {1, 4} weeks; crisis slices 2020 and 2022; edge_mode ∈ {scm, tyler}.  
- Ablations: gate strict vs soft; δ_frac grid around calibrated value; q_max ∈ {1,2,3}; off-component cap on/off.

### 8.2 Robustness / falsification checks (must-have)
- Synthetic non-Gaussian: elliptical / heavy-tail + mild time dependence calibration transfer study.  
  Anchor: `experiments/synthetic/calibrate_thresholds.py` + new mode (see `fjsprompt1output.md` task).
- Design validity diagnostics: quantify imbalance per window and correlate with detection/ΔMSE.  
  Anchor: `experiments/daily/grouping.py` + new artifact `design_diagnostics.csv`.

### 8.3 Output requirements (paper-facing)
- A single consolidated table per design: ΔMSE (EW/MV), detection/acceptance, turnover, and regime splits.
- ROC/FPR plots for calibration regimes actually used.
- “Where does overlay fire?” plots: edge margins, alignment histograms, rejection reasons.

---

## 9) Roadmap: paper-level (submission readiness)
Goal: a coherent story with defensible claims.

### 9.1 If results are positive (overlay helps somewhere)
- Formalize “when it helps” conditions:
  - prewhitening on/off
  - robust edge mode
  - calm vs crisis
  - window length
- Provide a theory-consistent simulation suite matching those conditions.

### 9.2 If results are negative (overlay harmful / unstable)
This is still publishable if framed correctly:
- “Selective de-aliasing is extremely sensitive to calibration and design violations; naive application fails.”
- Contribution: a careful audit with diagnostics + robust gating recommendations + evidence across regimes.

### 9.3 Nested design decision
- Only include nested if:
  - nested null FPR controlled,
  - nested coverage nonzero on real data,
  - nested adds incremental value beyond one-way.
Otherwise: “exploratory appendix.”

---

## 10) Commands & plots checklist (by milestone)

### Validity milestone (1–2 weeks)
Commands:
- `make test-fast`
- `EXEC_MODE=deterministic make sweep:acceptance`
- `PYTHONPATH=src python experiments/synthetic/nested_killtest.py --help` then run the kill-test config
- `EXEC_MODE=deterministic make rc-lite-sanity`

Plots/tables:
- `reports/figures/` ROC plots (one-way + any new modes)
- `reports/synthetic_nested_killtest/<RUN_ID>/` summary with null acceptance
- `reports/rc-*-sanity-*/`:
  - ΔMSE by design/regime
  - detection/acceptance + reason codes histogram
  - percent_changed (cap OFF)

### Experiment grid milestone (4–8 weeks)
Commands:
- `make rc` (or targeted grid runner) + `tools/make_summary.py`
- Ablations: `experiments/eval/sensitivity.py`, `experiments/ablate/run.py`

Plots/tables:
- consolidated CSV tables for ΔMSE, detection, turnover
- “overlay firing map” diagnostics by regime

### Paper milestone
- reproducible “paper run manifest” listing exact RUN_IDs, configs, and git SHAs
- frozen figures in `figures/paper/` and tables in `reports/paper/`

---
