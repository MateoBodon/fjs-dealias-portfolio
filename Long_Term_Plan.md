# LONG_TERM_PLAN.md — fjs-dealias-portfolio

Owner: Mateo Bodon  
Advisor: Prof. Zhou Fan  
Last updated: 2025-12-08  

Scope: FJS-style MANOVA de-aliasing overlay for high-dimensional covariance and portfolio risk estimation on equity return panels, calibrated via synthetic null/power and evaluated against shrinkage and factor-model baselines.

---

## 1. Vision and Main Research Questions

### 1.1 Overall vision

Use Fan–Johnstone–Sun (FJS) MANOVA-style de-aliasing as a *localized overlay* on top of standard covariance estimators for equity return panels, with the goal of:

- Improving **portfolio risk estimation** (variance, VaR, ES) in high-dimensional regimes (p comparable to n).
- Doing so in a way that:
  - Is **theory-backed** by FJS + El Karoui (risk underestimation) + factor models.
  - Is **transparent**: we know when and where the overlay is active.
  - Is **robust**: we never blow up risk in crises just because the detector misfires.

The overlay is not meant to replace shrinkage/factor estimators globally; it’s a regime- and window-specific correction layer that only triggers when the MANOVA detection says “there is an aliasing spike I understand and can safely adjust”.

### 1.2 Core estimands

At a high level:

- For a given rolling window \(t\), group design \(G\), and base covariance estimator \(\hat\Sigma_{\text{base}}\):
  - **De-aliased covariance** \(\hat\Sigma_{\text{FJS}}(t, G)\)  
    obtained by:
    1. Running an FJS-style MANOVA spike detection on group covariance contrasts.
    2. If accepted, applying a spectral transform \(T(\lambda)\) to isolate and shrink / reweight the detected spike directions.
    3. Otherwise, falling back to \(\hat\Sigma_{\text{base}}\).

- **Risk estimands**:
  - Realized out-of-sample **portfolio variance** for:
    - Equal-weight (EW) portfolio.
    - Constrained mean–variance (MV) portfolios (ridge + box + possibly turnover constraints).
  - **VaR** and **ES** at standard levels (e.g. 95%, 99%) based on next-day returns.

We care about:

- \(\Delta \text{MSE}_{\text{var}}(\hat\Sigma_{\text{FJS}}, \hat\Sigma_{\text{baseline}})\) on realized variance.
- VaR/ES **coverage** and **ES error**.
- **DM-style predictive tests** on loss series, especially restricted to the *flip set* (windows where overlay changes the covariance).

### 1.3 Main questions

1. **Detection correctness**  
   - Under realistic nulls (no true spike in the MANOVA sense), is the detection FPR controlled near the target (e.g. 2%) across:
     - Different groupings (DoW, nested year/week, volatility-state).
     - Different edge estimators (SCM, Tyler, Huber).
   - Under spiked alternatives, do we get nontrivial power with reasonable SNRs?

2. **Overlay value vs shrinkage and factor models**
   - For EW and constrained MV portfolios:
     - When the overlay fires, does it reduce variance MSE vs:
       - Ledoit–Wolf / OAS / robust SCM.
       - Observed-factor (FF5+MOM) and POET-lite covariances.
     - Is any gain robust across:
       - Calm vs crisis regimes.
       - Different universes (e.g. top 100 / 300 / 500 names, sector-balanced).

3. **Stability and safety**
   - Are there regimes (e.g., 2020 COVID crisis) where overlay systematically *harms* risk estimation even when detection is frequent?  
     If yes, can we throttle overlay (stricter gating, smaller transforms) to make it at least neutral?
   - How fragile is the nested design? When does it fail to accept windows, and is that fundamentally a data limitation or a guardrail/tuning issue?

4. **Positioning vs factor models**
   - Is the overlay just rediscovering factor structure, or is it catching residual spikes *after* prewhitening?
   - Do factor-based covariances + de-aliasing provide any incremental benefit vs factor-only or shrinkage-only baselines?

---

## 2. Estimators and Designs Being Compared

### 2.1 Base covariance estimators

These all live under `src/finance/` and related modules.

**Classical / shrinkage:**

- Sample covariance (SCM) and robust SCM variants (Tyler, Huber).
- Ledoit–Wolf, OAS (where implemented).
- Possibly RIE-style estimators if/when added (low priority unless already in repo).

**Factor-based:**

- Observed-factor covariance:
  - Prewhiten by FF5+MOM (or similar) factors.
  - Estimate covariance on residuals.
- POET-lite / approximate factor model estimators:
  - Top-k PCs as factors + sparse idiosyncratic covariance.

### 2.2 FJS overlay

- Detection step:
  - Group design: partition weekly returns into groups (e.g., days-of-week, volatility states, nested year ⊃ week).
  - For each rolling window:
    - Compute MANOVA-like contrasts across groups.
    - Estimate the MP edge using SCM/Tyler/Huber.
    - Compute the *edge margin* and an acceptance statistic based on the FJS surrogate \(T(\lambda)\).
  - Gating:
    - Only accept if:
      - Edge margin > threshold.
      - Isolation/stability checks pass.
      - Synthetic-calibrated null FPR stays near target.

- Overlay step (when accepted):
  - Decompose \(\hat\Sigma_{\text{base}} = Q \Lambda Q^\top\).
  - Apply transformation \(T(\Lambda)\) only along the detected spike subspace.
  - Reconstruct \(\hat\Sigma_{\text{FJS}} = Q T(\Lambda) Q^\top\).
  - Respect guardrails:
    - Condition number caps.
    - Upper/lower bounds on resulting variances.

### 2.3 Grouping / design options

Implemented or planned designs in `experiments/equity_panel/`:

- **DoW (one-way)**  
  Groups: Monday–Friday, using weekly windows, capturing day-of-week effects.

- **Nested year ⊃ week**  
  Two-way structure:
  - Year as outer group, week-of-year as inner repeated structure.
  - Goal: capture persistent yet structured temporal heterogeneity in spikes.

- **Volatility-state design**  
  Groups determined by regime (e.g., low/medium/high realized volatility states).

- **Synthetic designs**  
  In `experiments/synthetic/` for null/power calibration:
  - Gaussian and elliptical spiked models.
  - Variable group structures to emulate DoW and nested designs.

### 2.4 Portfolio constructions

Implemented in `experiments/eval/` + `src/evaluation/`:

- Equal-weight (EW) portfolios: simple baseline, sensitive to covariance scale.
- Constrained mean–variance (MV):
  - Target volatility or return with:
    - Ridge penalty on weights.
    - Box constraints (per-asset exposure).
    - Optional turnover constraint or penalty.

---

## 3. Workstreams

We group work into five interacting workstreams.

### 3.1 Theory alignment

Goals:

- Keep the implementation anchored in:
  - FJS random-effects MANOVA theory for spike detection and de-aliasing.
  - El Karoui’s analysis of high-dim Markowitz risk underestimation.
  - Factor model covariance literature for baselines (Fan–Fan–Lv, etc.).

Concrete tasks:

- Maintain a short `docs/THEORY_NOTES.md` with:
  - Mapping from FJS notation to code variables (e.g., how \(T(\lambda)\) appears in `src/fjs/`).
  - Where and how MP edges are estimated (SCM vs Tyler vs Huber).
  - Conditions under which the overlay should be close to identity (i.e., no or weak spikes).
- Maintain a `docs/THEORY_CHECKLIST.md` you can walk through before any major change to detection/overlay logic.

### 3.2 Synthetic harness

Goals:

- Provide a controlled environment to:
  - Estimate null FPR for each design and edge mode.
  - Explore power and robustness across spike strengths, group structures, and noise models.

Concrete tasks:

- Keep `experiments/synthetic/` up to date:
  - `null.py` — null simulations for each design and edge mode.
  - `power.py` — spiked alternatives with configurable SNR.
  - `calibrate_thresholds.py` — sweep acceptance/gating hyperparameters to hit target FPR.
- Produce and version:
  - ROC curves and acceptance-rate tables.
  - `calibration/edge_delta_thresholds.json` and friends.
- Ensure there is always a **calibration manifest** (JSON/YAML) that records:
  - Git commit.
  - Seed / trial count.
  - Designs and edge modes.
  - Chosen thresholds.

### 3.3 Real-data evaluation

Goals:

- Evaluate overlay vs baselines on WRDS-style daily equity panels:
  - DoW, nested, vol-state designs.
  - Calm vs crisis periods.
  - EW and MV portfolios.

Concrete tasks:

- Maintain a small suite of **smoke configs**:
  - `config.smoke.yaml` — small asset universe, short time span.
  - `config.rc-lite.yaml` — moderate asset universe, limited windows, used for “sanity RC”.
- Maintain **full RC configs**:
  - `config.rc.yaml` — long span, larger universes.
  - Crisis configs (`config.crisis.2020.yaml`, `config.crisis.2022.yaml`, etc.).
- For each RC, generate:
  - Rolling metrics CSVs (`metrics_summary.csv`, `detection_summary.csv`, `dm_results.csv`, etc.).
  - Figures under `figures/rc/YYYYMMDD/`.
  - Memo + brief under `reports/rc-YYYYMMDD/`.

### 3.4 Infra/tooling

Goals:

- Make it easy to run and reproduce experiments on both laptop and Hetzner.

Concrete tasks:

- Make targets:
  - `make test-fast`, `make test-integration`, `make test-slow`, `make test`.
  - `make sweep:acceptance`, `make calibrate-thresholds`.
  - `make rc`, `make rc-lite`, `make rc-lite-sanity`, `make rc-crisis`.
- Tools:
  - `tools/build_gallery.py`, `tools/build_memo.py`, `tools/build_brief.py` to assemble RC artifacts.
  - `tools/prewhiten_effect.py`, `tools/summarize_run.py` for diagnostics.
- Cloud / HPC docs:
  - `docs/HPC.md` describing how to run heavy jobs on Hetzner with WRDS data mounted.

### 3.5 Documentation & project-state

Goals:

- Keep human-facing docs and project-state markdown aligned with code.

Concrete tasks:

- Maintain `PROJECT_STATE/` (or `docs/PROJECT_STATE/`) with:
  - `PIPELINE_FLOW.md` — ingest → designs → estimators → overlay → portfolios → metrics.
  - `DATAFLOW.md` — where data lives, what intermediates are produced.
  - `ARCHITECTURE.md` / `MODULE_OVERVIEW.md` — module-level overview.
  - `EXPERIMENTS.md` — grid of experiments, with status.
  - `CURRENT_RESULTS.md` — readable summary of what’s working.
  - `OPEN_QUESTIONS.md`, `KNOWN_ISSUES.md`, `ROADMAP.md`.
  - `CONFIG_REFERENCE.md`, `SERVER_ENVIRONMENT.md`, `TEST_COVERAGE.md`.
- Maintain `CHANGELOG.md` and `docs/AGENT_RUNS/*.md` for structural changes and Codex sprints.

---

## 4. Milestones and Horizons

### 4.1 Short-term (1–2 weeks)

Theme: **Stabilize detection + “sanity RC” path.**

Targets:

1. **AGENTS + infra hygiene**
   - `AGENTS.md` present and correct.
   - `docs/HPC.md` explains Hetzner setup and recommended commands.
   - `make test-fast` clean on laptop and Hetzner.

2. **Nested detection out of “zero-coverage” mode**
   - Identify the nested config(s) with 0% accepted windows.
   - Instrument additional diagnostics (per-year sample sizes, isolation checks).
   - Tune guardrails so that:
     - Synthetic nested null FPR ~2%.
     - Real nested smoke runs show 2–6% detection coverage.

3. **RC-lite sanity pipeline**
   - Implement `make rc-lite-sanity`:
     - Uses small time span and small universe.
     - Runs DoW + vol-state (and optionally nested) designs.
   - Generates:
     - Minimal `metrics_summary.csv`, `detection_summary.csv`.
     - A small gallery + brief under `reports/rc-YYYYMMDD/`.

Deliverable: one **sanity RC drop** that you can send to Fan as “we can reproduce this overnight and it matches the description in LONG_TERM_PLAN”.

### 4.2 Medium-term (4–8 weeks)

Theme: **Full experimental grid, crisis vs calm, factor vs shrinkage vs overlay.**

Targets:

1. **Calibrated overlay across designs**
   - Null/power calibration for DoW, nested, vol-state designs and multiple edge modes.
   - FPR near target across designs; power curves for spike SNRs.

2. **Crisis tuning**
   - Focused crisis RCs (2020, 2022) with overlay throttled to “safe”:
     - Overlay no longer obviously worse than shrinkage.
     - Ideally neutral in crisis, possibly modest improvements in calm periods.

3. **Factor vs shrinkage vs overlay comparisons**
   - On a fixed set of panels:
     - Compare EW and MV risk metrics for:
       - Shrinkage-only.
       - Factor-only (FF5+MOM, POET-lite).
       - Overlay-on-shrinkage.
       - Overlay-on-factor (if meaningful).
   - Summarize results in `CURRENT_RESULTS.md` and RC memos.

4. **Ablation grid**
   - Complete a tractable ablation grid:
     - Key hyperparameters: edge mode, gating thresholds, q_max, overlay strength.
     - Designs: DoW + at least one of nested / vol-state.
   - Visual summary: ablation heatmap in gallery and memo.

Deliverable: a **full RC drop** + memo that tells a coherent story about when overlay helps, is neutral, or hurts vs shrinkage and factor baselines.

### 4.3 Long-term / paper-level

Theme: **Turn this into a publishable paper or workshop-style writeup.**

Targets:

1. **Lock in a primary design + estimator comparison**
   - Choose one or two designs (e.g., DoW + nested) and a small set of estimators:
     - SCM + overlay.
     - Ledoit–Wolf/OAS.
     - Factor covariance (FF5+MOM and/or POET-lite).
   - Define the main comparison table you’d show in a paper.

2. **Robustness & sensitivity**
   - Additional datasets or universes:
     - Different time periods (e.g., 1990s vs 2010s).
     - Alternative universes (e.g., S&P 500 vs sector-balanced).
   - Robustness to:
     - Window length.
     - Asset count (p).
     - Factor set.

3. **Theory integration**
   - Short theory section:
     - Explain how the FJS MANOVA detection maps to the equity-panel design.
     - Show how El Karoui’s risk underestimation factor manifests in your simulations/empirics.
     - Explain factor baselines.

4. **Repo hardening**
   - Environment capture (e.g. `environment.yml`, `pyproject.toml` pinned).
   - One-command repro script:
     - Downloads WRDS data (given creds).
     - Runs calibrations + RC configs.
     - Produces the tables/figures used in the paper.

Deliverable: **Paper-ready results** and a hardened repo that a referee or collaborator can run.

---

## 5. Experiment Grid

### 5.1 Core experiments (must-have)

These should be run and re-runnable:

1. **Synthetic null/power calibration**
   - Designs: DoW, nested, vol-state.
   - Edge modes: SCM, Tyler (Huber optional).
   - Outputs:
     - ROC curves and FPR vs threshold curves.
     - Chosen thresholds with justification.

2. **DoW and vol-state equity panels**
   - Universes: top-100, top-300 by market cap.
   - Periods:
     - Long-span (e.g., 2000–2024).
     - Crisis-focused slices (e.g., 2008–09, 2020, 2022).
   - Estimators:
     - SCM, LW/OAS, robust SCM.
     - Factor obs, POET-lite.
     - Each with/without overlay.
   - Portfolios:
     - EW, MV (ridge + box).

3. **Nested design on equities**
   - At least one long span and one crisis slice.
   - Same estimator set as DoW, but smaller grid if needed.
   - Goal: demonstrate that nested overlay occasionally triggers and does not obviously degrade risk.

4. **Daily evaluation harness**
   - On a subset of the above:
     - EW and MV daily portfolios.
     - Realized variance, VaR/ES, DM tests.

### 5.2 Nice-to-have experiments

These are lower priority; do them if core results look promising and time permits:

- Additional designs:
  - Sector-based groups (e.g., GICS sectors) as group labels.
- Alternative universes:
  - Small-cap vs large-cap splits.
  - Regions (US vs global).
- Extra estimators:
  - RIE / nonlinear shrinkage variants.
  - Other robust covariance estimators.
- Factor variants:
  - Alternative factor sets (e.g., quality, low-vol) if data accessible.

---

## 6. Risks and Kill-Switch Tests

### 6.1 Conceptual risks

1. **Overlay adds little to strong shrinkage/factor baselines**
   - If across panels and regimes:
     - ΔMSE and VaR/ES metrics show overlay as mostly neutral.
     - Gains do not survive simple robustness checks.
   - Then the method may not warrant a standalone paper vs existing shrinkage/factor techniques.

2. **Detection is too fragile / data design mismatch**
   - Nested design might fail systematically (0% accepted).
   - Vol-state or DoW designs might be too weak to reveal FJS-like spikes in real data.

3. **Crisis harm**
   - In crisis periods, overlay may systematically *underestimate* risk or distort portfolio allocation.

4. **Scope creep and unmanageable complexity**
   - Too many designs, estimators, and hyperparameters, making it hard to tell a clear story.

### 6.2 Kill-switch tests

These are tests you can run early and often:

1. **Kill-switch: synthetic null FPR**
   - If for any design/edge mode your estimated null FPR sits consistently above, say, 5–10% at your chosen threshold, and you cannot fix it via modest threshold changes, that configuration is out.

2. **Kill-switch: crisis ΔMSE**
   - If in 2020 crisis:
     - Overlay harms MV risk by a large margin vs shrinkage/factor baselines *even after* throttling (e.g., stricter gating, smaller transforms).
   - Then either:
     - Restrict the method to calm regimes, or
     - Narrow the claims to “diagnostic overlay” rather than performance improvement.

3. **Kill-switch: coverage vs usefulness**
   - If after tuning, overlay:
     - Fires <1–2% of windows (too rare) *and* the few flips do not show meaningful ΔMSE or allocation changes.
   - Then the design might not be suitable; deprioritize it.

4. **Kill-switch: factor redundancy**
   - If prewhitened datasets show that overlay essentially re-creates factor PCs and does not improve risk vs factor covariance alone, reposition the framing as:
     - “Overlay reveals factor-like structure” rather than “improves risk estimation”.

---

## 7. Documentation Policy and Project-State Updates

### 7.1 General principles

- **Docs are part of the deliverable**, not an afterthought.
- Whenever a structural change happens (new design, new estimator, changed gating), **code + docs + configs must move together**.
- Codex and human edits should always:
  - Update `CHANGELOG.md`.
  - Update relevant `PROJECT_STATE/*.md` files.
  - Log the work in `docs/AGENT_RUNS/<date>_codex_sprint_<N>.md`.

### 7.2 Required docs

- `docs/LONG_TERM_PLAN.md` (this file)  
  High-level vision, experiment grid, and milestones.

- `PROJECT_STATE/` (or `docs/PROJECT_STATE/`), including:
  - `PIPELINE_FLOW.md` — current pipeline diagram.
  - `DATAFLOW.md` — where data flows and where it’s cached.
  - `EXPERIMENTS.md` — a table of configs, design types, and status (planned/running/done).
  - `CURRENT_RESULTS.md` — plain language summary of key findings.
  - `OPEN_QUESTIONS.md`, `KNOWN_ISSUES.md` — active questions and bugs.
  - `ROADMAP.md` — shorter term plan; link back to LONG_TERM_PLAN.
  - `CONFIG_REFERENCE.md`, `SERVER_ENVIRONMENT.md`, `TEST_COVERAGE.md`, `STYLE_GUIDE.md`.

- `docs/HPC.md`  
  Hetzner instructions, including:
  - Paths for WRDS data.
  - Recommended commands for calibrations and RCs.

- `AGENTS.md`  
  For Codex and other agents (separate file; see below).

### 7.3 Update cadence

- After every **RC**:
  - Update `CURRENT_RESULTS.md` with key metrics and takeaways.
  - Update `EXPERIMENTS.md` (status + links to RC folders).
  - Append to `ROADMAP.md` noting which items were completed and which rolled over.

- After every **structural code change**:
  - Update relevant architecture/flow docs.
  - Note the change in `CHANGELOG.md`.

- After every **Codex sprint**:
  - New or updated `docs/AGENT_RUNS/<date>_codex_sprint_<N>.md`:
    - Goals, plan, commands run, tests, outcomes, remaining questions.

---

## 8. Advisor-Facing Story if Things Work

If everything goes reasonably well, the story you want to tell Zhou Fan is:

1. **We implemented an FJS-style de-aliasing overlay as a MANOVA-informed spectral transform on equity panels.**  
   - Detection uses group designs (DoW, nested, vol-state) and MP edge estimates (SCM/Tyler).
   - Overlay only triggers when the synthetic-calibrated detector says “this is a credible spike”.

2. **On synthetic data, we can control null FPR and characterize power across designs and edge modes.**  
   - ROC curves show that for realistic spike strengths, designs like DoW and nested have nontrivial power.
   - Thresholds are chosen to keep FPR ~2%, and we can show how that was calibrated.

3. **On real equity panels, the overlay is mostly neutral but sometimes helpful compared to shrinkage/factor baselines.**  
   - For EW and constrained MV portfolios:
     - ΔMSE and DM tests show that in some regimes (especially calm or moderately volatile), overlay reduces variance error or improves VaR/ES coverage.
     - In crisis regimes, a throttled overlay is at least not worse than shrinkage-only, and in some specs it remains slightly helpful.

4. **Factor models remain strong baselines, but overlay adds a distinct angle.**  
   - Prewhitening + overlay sometimes finds residual spikes that are not fully explained by FF5+MOM/POET.
   - Where overlay doesn’t help beyond factors, we can say so and restrict claims to certain regimes.

5. **We have a reproducible end-to-end pipeline.**  
   - Given WRDS access, a single `make rc` (plus calibration) regenerates:
     - Synthetic FPR/power tables.
     - Real-data RCs with figures and memos.
   - The repo has clear docs (`LONG_TERM_PLAN.md`, `AGENTS.md`, PROJECT_STATE files) and can be handed to collaborators or a referee.

6. **If we decide to write, we already have a natural structure.**  
   - Section 1–2: theory recap (FJS, El Karoui, factor models).
   - Section 3: data and designs.
   - Section 4: synthetic calibration.
   - Section 5: real-data results (DoW, nested, vol-state; EW & MV; crisis vs calm).
   - Section 6: discussion of when overlay is useful vs when shrinkage/factors suffice.

This is the bar that justifies turning the work into a real paper rather than “just a neat implementation”.
