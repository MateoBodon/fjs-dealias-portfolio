# Plan of Record — FJS De-aliasing Overlay for Portfolio Risk

Last updated: 2025-12-19  
Primary “ground truth” references:
- `project_state/PIPELINE_FLOW.md`, `ARCHITECTURE.md`, `DATAFLOW.md`, `CURRENT_RESULTS.md`, `KNOWN_ISSUES.md`, `OPEN_QUESTIONS.md`, `ROADMAP.md`
- `PROGRESS.md` (repo root)

## 0) Non-negotiables (stop bullsh*t early)
- No claims based on:
  - capped/truncated evaluations (window caps, condition caps) without explicit labeling + sensitivity checks
  - silent fallbacks (especially portfolio solver fallbacks)
  - “acceptance=0” designs where the overlay never actually applies
  - diagnostics that lump failures into `guard_other` or `diagnostic_failure` without actionable attribution
- Every code change must be traceable to:
  - a run log in `docs/agent_runs/<RUN_NAME>/` and
  - passing tests (`make test-fast` at minimum)
- “Pipeline valid” is a prerequisite for “research result”. If pipeline validity fails, we stop and fix validity.

## 1) Research framing (what we are actually trying to learn)

### 1.1 Estimand(s)
We care about out-of-sample risk forecasting and portfolio risk, not “pretty spectra”.

For each rolling window t:
- Input: returns matrix \(R_t \in \mathbb{R}^{T \times p}\) (after optional factor prewhitening).
- Build covariance estimate \(\hat\Sigma_t\).
- Construct portfolio weights \(w_t\) under a fixed portfolio rule (EW and constrained MV).
- Observe forward realized risk over horizon h (e.g., 21d):
  - realized variance \( \hat\sigma^2_{t+h}(w_t) \)
  - tail metrics (VaR/ES) computed from forward returns.

Primary estimands:
- **ΔLoss** between treatment and baseline on aligned windows:
  - ΔMSE (variance forecast error)
  - ΔQLIKE (variance forecast QLIKE)
  - ΔVaR/ΔES errors (if implemented consistently)
- **Operational impact**:
  - detection rate / acceptance rate
  - weight change statistics (turnover, % changed, etc.)
  - skip rates (conditioning cap, solver missing, balance failure)

### 1.2 Treatment vs baseline (what changes)
Treatment = **FJS/MANOVA de-aliasing overlay** applied to a base covariance estimate, gated by detection diagnostics.

Implementation anchors:
- Detection / gating / overlay: `src/fjs/overlay.py`, `src/fjs/gating.py`, `src/fjs/mp.py`
- Robust edge modes: `src/fjs/robust.py`
- Portfolio solver + explicit skip/fail-loud semantics: `src/finance/portfolios.py`

Define:
- Base covariance (baseline): \(\hat\Sigma^{base}_t\)
- Treatment covariance: \(\hat\Sigma^{treat}_t = \text{Overlay}(\hat\Sigma^{base}_t, \text{detect\_spikes}(R_t, g_t), \text{cfg})\)
- Gating: if guardrails fail, treatment collapses to baseline (but must be logged as “no-op due to guard reason X”, not silently).

Baselines we must compare against (minimal set):
- SCM + shrinkage baselines: Ledoit–Wolf / OAS (`src/finance/ledoit.py`)
- RIE/QUEST shrinker (`src/finance/rie.py`)
- Robust scatter (Tyler/Huber) as the MP edge mode / base (`src/fjs/robust.py`)
- Factor prewhitening toggle (FF5+MOM) as a controlled ablation (`src/eval/metrics.py` + factor utilities)

### 1.3 Data design (what statistical structure is assumed)
Repo supports:
- Daily evaluation runner: `experiments/eval/run.py`
  - group designs: `dow`, `vol`, `week`, `dowxvol`
- Weekly equity panel runner: `experiments/equity_panel/run.py`
  - designs: `oneway`, `dow`, `vol`, `nested`
  - optional `--gating-diagnostics` emits per-window guardrails

Data sources (current default):
- returns: `data/returns_daily.csv` (see `project_state/DATAFLOW.md`)
- factors: `data/factors/ff5mom_daily.csv`

“Publishable” designs (minimum viable):
- **Daily DoW** (one-way balanced grouping) — works today but effect is currently weak/negative in sanity summaries.
- **Daily vol-state** — currently balance is fragile; must be made robust or dropped.
- **Nested weekly** calibration refreshed (Dec 20 2025): synthetic null-FPR now ≤2% (0/220, CI hi 0.017) with full power at delta_frac=0.05; still optional for paper v1 until real-data acceptance is rechecked.

## 2) Minimal assumptions for the theory to apply (and whether we respect them)

### 2.1 Minimal assumptions (what we need, not what we wish)
For MP-edge + spike detection + de-aliasing to be meaningful:
- High-dimensional regime: p and T of similar order (p/T not tiny).
- Returns (after optional factor prewhitening) are approximately:
  - mean-zero
  - weakly dependent (windowed dependence tolerated, but must be checked via robustness)
  - approximately elliptical / sub-Gaussian is “good enough” for empirical MP-ish behavior
- Balanced design:
  - one-way: equal replicates per group
  - nested: equal reps per subgroup; correct labeling (e.g., year/week)
- Calibration: gating thresholds chosen to target a null-FPR (synthetic harness), not tuned on the same real data being evaluated.

### 2.2 Where the repo matches vs violates assumptions (current)
Matches / partially matches:
- Balanced-window construction exists (`src/eval/balance.py`) and “balance failure” is logged.
- Factor prewhitening exists and is measurable (telemetry).
- Synthetic harness exists for FPR/power calibration (`experiments/synthetic/*`).

Violations / high-risk approximations:
- Time dependence is real and strong; we rely on windowing + robust estimators + factor prewhitening, but we must quantify sensitivity (block bootstrap / regime slicing).
- Nested design currently shows high null FPR + near-zero acceptance (KNOWN_ISSUES). That means we do *not* have valid control over false discovery in the one place where “MANOVA” is most central.

## 3) Minimal publishable package (what we must deliver)

### 3.1 A “paper v1” that is actually defensible
Minimum contribution:
1) A **well-instrumented**, calibrated de-aliasing overlay pipeline with:
   - explicit gating reasons (no `guard_other`)
   - explicit skip reasons (no silent drop/fallback)
   - validated synthetic null-FPR control
2) A **small but clean** real-data study showing either:
   - improvement in out-of-sample risk losses in at least one regime/design (with stability checks), OR
   - a crisp negative result: “overlay does not help under realistic dependence; gating prevents harm; improvement requires XYZ conditions”, backed by controlled ablations.

### 3.2 Minimal experimental grid (do not expand before this is clean)
Real data (daily runner `experiments/eval/run.py`):
- Designs: `dow` and `vol`
- Universe: top-60 (or top-50 if that’s the current stable sanity default)
- Window/horizon: 126×21 (current norm)
- Prewhitening: off vs FF5+MOM on (paired)
- Estimators:
  - baseline: `oas`, `lw`, `quest` (RIE)
  - treatment: `dealias` overlay on top of a fixed base + edge-mode (Tyler vs SCM)
- Portfolios:
  - EW (sanity)
  - MV constrained (box + ridge + turnover; BUT must log when constraints bind / when cap skips happen)

Synthetic validation:
- One-way null + power:
  - target null-FPR = 2% (or whatever current calibration target is)
  - show power curves vs μ
- Nested null “kill-test”:
  - must demonstrate null-FPR control before we claim anything nested

## 4) Acceptance criteria for “pipeline is valid” (gate to proceed)

A run is “valid for research conclusions” only if ALL are true:

### 4.1 Code / tests
- `make test-fast` passes.
- Any changed behavior has a unit test or integration test covering it.

### 4.2 Data integrity
- `tools/verify_dataset.py` passes for every dataset used (and hashes are recorded in run metadata).

### 4.3 No silent fallbacks
- Portfolio optimization:
  - missing solver is either fail-loud OR explicit skip with `skipped=true` and reason; never “equal-weight fallback”.
- No silent window dropping:
  - any skip due to condition cap / infeasible optimizer / balance failure must appear in diagnostics and summary tables.

### 4.4 Diagnostics are attributable
- Weekly and daily gating outputs must not contain:
  - `guard_other` unless it is provably unreachable, OR it includes a structured `guard_detail` explaining which guard fired.
  - `diagnostic_failure` without a stack/exception class + minimal context.

### 4.5 Calibration validity
- Synthetic null-FPR is at or below target for every mode we claim (edge_mode × design).
- Calibrated thresholds are versioned and referenced by path + git SHA in run metadata.

## 5) Roadmap

### Horizon 1: 1–2 weeks (debug + validity + advisor-ready RC run)
Goal: get to a state where **one** RC-lite run can be defended without embarrassment.

1) Fix weekly gating attribution (remove `guard_other` dominance)
- Code: `experiments/equity_panel/run.py` (especially `_infer_skip_reason(...)` and gating diagnostics writer)
- Tests: `tests/experiments/test_gating_diagnostics.py` (+ add regression test: no `guard_other`)
- Command (smoke):
  - `EXEC_MODE=deterministic make run:equity_smoke`
  - plus one direct weekly run with `--gating-diagnostics` using `experiments/equity_panel/config.smoke.yaml`
- Plots/tables:
  - `weekly_diagnostics.md` must include: top reasons + examples
  - `gating_diagnostics.csv` must include: reason codes + key stats

2) Nested null-FPR triage (do not “tune until it looks good”)
- Run:
  - `python -m experiments.synthetic.nested_killtest --config <nested-killtest-config> --out reports/synthetic/nested_killtest/<RUN_ID>`
- Output:
  - markdown summary + FPR table by threshold
- Decision:
  - If we cannot hit null-FPR ≤ target without killing all power, nested is postponed for paper v1.

3) Evaluation contamination checks (caps/selection bias)
- Daily runner must report:
  - skip shares per reason
  - aligned window counts for DM tests (`n_effective`)
  - whether comparisons are on intersection sets
- Code hotspots:
  - `experiments/eval/run.py` (alignment + skip policy)
  - `tools/make_summary.py` / `tools/summarize_rc_sanity.py` (do not mix capped and uncapped without labels)
- Plots/tables:
  - a “skip reason histogram” per design
  - a “cap binding rate” table (if we can compute)

4) Produce one advisor-ready run with strict validity
- Command:
  - `EXEC_MODE=deterministic make rc-lite-sanity`
  - Then: `PYTHONPATH=src:. python3 tools/make_summary.py --rc-dir <rc-dir>`
- Required outputs:
  - `summary/summary_perf.csv`
  - `summary/summary_detection.csv`
  - `summary/limitations.md` (auto-generated, but must mention any caps/skips)
  - `summary/completeness.json` (must say complete + uncapped)
- Update:
  - `project_state/CURRENT_RESULTS.md` and `PROGRESS.md` with run IDs + deltas

### Horizon 2: 4–8 weeks (full experiment grid + robustness)
Goal: enough evidence for a coherent draft.

1) Full daily grid (real data)
- Designs: `dow`, `vol`, (optional `week`, `dowxvol` if stable)
- Prewhitening: off vs on (paired)
- Edge modes: `scm`, `tyler`
- Shrinkers: `lw`, `oas`, `quest`
- Gate regimes:
  - strict vs soft (if both exist)
  - calibrated delta_frac vs fixed delta_frac sensitivity
- Deliverables:
  - grid summary tables (ΔLoss + DM tests + acceptance/skip)
  - regime-sliced results (calm vs crisis)

2) Crisis safety checks
- Must show overlay does not systematically degrade during crisis windows.
- Require:
  - explicit guardrails that reduce acceptance in crisis if unstable
  - crisis-only table of ΔQLIKE / ΔES errors + skip rates

3) Only if nested is fixed: nested real-data evaluation
- First: nested null-FPR synthetic passes.
- Then: weekly nested on real data must have:
  - acceptance within a sane band (2–6% target from ROADMAP/KNOWN_ISSUES)
  - attributable skip reasons (not “other”)

### Horizon 3: Paper-level (submission readiness)
Goal: referee-proof story and reproducible artifacts.

- Lock a “paper configuration”:
  - pinned dataset hashes
  - pinned calibration JSON(s)
  - pinned config YAMLs
- Add simulations that match violations of assumptions:
  - dependence (AR/vol clustering)
  - factor mis-specification
  - heavy tails
- Add ablations that isolate mechanism:
  - effect of prewhitening alone
  - effect of robust edge alone
  - overlay on/off holding everything else fixed
- Produce camera-ready:
  - main tables (ΔQLIKE/ΔMSE/ΔES), acceptance/skip, FPR/power
  - appendix plots (ROC curves, gating diagnostics distributions, sensitivity)

## 6) Reporting outputs we will standardize (must exist for every “valid” run)
- Run metadata:
  - `run.json` + `resolved_config.{json|yaml}` in the run directory
  - dataset hash list
  - git SHA + dirty flag
- Core tables:
  - `summary_perf.csv` (loss deltas + DM)
  - `summary_detection.csv` (detection/acceptance/skip)
  - `limitations.md` (auto text listing caps/skips/known failure modes)
- Diagnostics:
  - daily: `diagnostics.csv`, `diagnostics_detail.csv`
  - weekly: `gating_diagnostics.csv`, `weekly_diagnostics.md`

## 7) Decision gates / pivot triggers (be explicit)
- If after fixing diagnostics + calibration, the overlay is:
  - acceptance >0 but ΔLoss is consistently ≥0 (harmful) across designs,
  - and this is robust to reasonable ablations (prewhitening, edge_mode),
  => pivot framing to: “de-aliasing is fragile under time dependence; gating is required; conditions for benefit are X”.
- If nested null-FPR cannot be controlled without killing power,
  => drop nested from paper v1; keep as “future work” with a documented failure analysis.
