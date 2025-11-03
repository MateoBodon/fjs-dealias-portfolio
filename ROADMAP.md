# ROADMAP.md (proposed)

> This plan is grounded in your existing **Vision** (replicates via Day-of-Week / volatility-state, robust MP-edge, calibrated gating, conditional overlay; success/kill criteria) and your **Extended Proposal** / **Memo** (FJS Algorithm-1 de-aliasing, guardrails, evaluation) so we preserve continuity while adding concrete deliverables and checks.     

## 0) North Star

* **Goal.** Implement and evaluate an **FJS de-aliasing overlay** for portfolio risk: detect aliased spikes near the MP edge, correct **eigenvalues only**, and blend with strong shrinkage/factor baselines; be **helpful in crises, neutral otherwise**; ship a clean, reproducible memo + gallery.  
* **Success criteria.** In high-volatility regimes, **ΔMSE ≥ 3–5%** and better VaR/ES vs best baseline; neutral in calm; **FPR ≤ 1–2%** from synthetic calibration at matched (p/n). If not achieved after calibration → publish a principled **negative result** with diagnostics.  

## 1) Architecture & Repo Layout

* `src/`

  * `fjs/` — MP edge/roots, admissible-branch checks, surrogate (t(λ,a)), de-aliasing search & substitution, robust edges (Tyler/Huber). (exists conceptually; keep modular) 
  * `baselines/` — **LW, OAS, CC** (have); **RIE/QuEST** (add); **EWMA** (add); **factor_obs** (FF5+MOM) & **POET** (unobserved). 
  * `overlay/` — overlay policy that swaps de-aliased eigenvalues in accepted directions and shrinks the rest. 
  * `eval/` — rolling windows, regimes (Crisis/Calm/Full), DM tests, VaR/ES, diagnostics. 
  * `synthetic/` — null/power generators at matched (p/n); produces ROC, δ/δ_frac tables. 
* `experiments/`

  * `daily/` — **DoW** and **Vol-state** designs; observed-factor **prewhitening** pipeline. 
  * `etf_panel/` — cleaner country/sector ETF demo (optional alt-panel). 
* `reports/` — `rc-YYYYMMDD/` with CSVs, plots, `memo.md`, `overlay_toggle.md`, and `summary.json`. 
* `tests/` — unit/integration tests for MP edge, gating, overlay, baselines, and eval metrics.
* `ROADMAP.md`, `AGENTS.md`, `CHANGELOG.md`, `RUNBOOK.md`.

## 2) Data Definition (make MANOVA “happy”)

* **Estimation frequency:** use **daily** returns for estimation; keep weekly only for reporting if desired. 
* **Groupings:**

  * **Primary:** Day-of-Week (Mon–Fri) one-way (many replicates).
  * **Alt:** **Volatility-state** (VIX/realized-vol terciles/quartiles) to stress regimes. 
* **Hygiene:** survivorship-clean, liquid universe with continuous coverage; winsorize daily tails; robust scatter for edge. 

## 3) Detection & Calibration

* **Edges:** robust MP edge on SCM & Tyler/Huber scatters; log **edge margins** and angle diagnostics. 
* **Guardrails:** **admissible branch** (z'(m)>0), **edge buffer** δ (abs+relative), **isolation**, **angular stability** across nearby (a), **q-discipline**, **null discipline** via FPR control. 
* **Calibration:** run **null/power** sims at matched (p/n) to pick δ / δ_frac / stability thresholds targeting **FPR ≤ 1–2%** and power at (μ∈{4,6,8}). 

## 4) Overlay & Baselines

* **Overlay policy:** if gate accepts, **replace eigenvalues** (λ → μ = λ/t_r); **preserve directions**; shrink remainder via best baseline. 
* **Baselines set:** LW, OAS, CC, **RIE/QuEST (add)**, **EWMA (add)**, **observed factors** (FF5+MOM), **POET**. 

## 5) Evaluation

* **Rolling:** daily estimation windows (n), horizon (H).
* **Regimes:** Crisis (high-VIX) / Calm / Full.
* **Portfolios:** EW + box/ridge-regularized MV (add turnover constraint if needed).
* **Metrics:** Variance **MSE**, **VaR(95%)**, **ES**, **DM tests**, and detection telemetry (rate, isolation share, edge margin, stability).  

## 6) Reproducibility & Docs

* **RC cadence:** every **1–2 weeks**; tag `rc-YYYYMMDD`; archive artifacts under `reports/rc-*`. 
* **Memos/Gallery:** one-command refresh to regenerate memo tables/plots and a small gallery. 
* **CHANGELOG:** conventional commits summarized per RC.

## 7) Research Framing (why this is the right approach)

* **Finance motivation:** plug-in Markowitz **underestimates risk** at high (p/n) (≈ (1-p/n) effect), worse under **elliptical** tails ⇒ strong baselines + robust edges are essential. 
* **Factor baselines & prewhitening:** observed-factor covariance reduces dimensionality; also a natural prewhitening step. 
* **Weak factors caution:** PC directions can be fragile under weak factors ⇒ prefer eigenvalue correction + stability-screened directions. (Positioning consistent with FJS + factor asymptotics.) 

## 8) Phased Timeline (12–16 weeks)

**Phase A — Calibration & Daily Designs (Weeks 1–3)**

* Implement **DoW** & **Vol-state** groups; robust edge defaults; null/power calibration; deposit `calibration/*.json` thresholds; smoke runs. **Deliverables:** calibration tables, ROC, detection dashboard.

**Phase B — Overlay & Baselines (Weeks 4–6)**

* Integrate **RIE/QuEST** + **EWMA**; add **observed-factor prewhitening** toggle; implement overlay with diagnostics. **Deliverables:** unit tests; ablation tables.

**Phase C — Regimes & RC (Weeks 7–9)**

* Crisis/Calm/Full runs; ΔMSE/DM/VaR/ES with detection telemetry; `rc-YYYYMMDD` artifacts + memo draft.

**Phase D — Alt-Panels & Paper (Weeks 10–12)**

* ETF demo; finalize memo/paper; **negative-result** track if overlay neutral. **Deliverables:** camera-ready memo, gallery.

**Phase E — Polishing (Weeks 13–16)**

* CI hardening, docs, reproducibility checks; final RC + tag.

## 9) Risks & Mitigations

* **Sparse detections:** use daily replicates, robust edges, calibrated thresholds; overlay neutral when off. 
* **Baseline dominance:** constrain claim to **crises**; negative result acceptable with diagnostics. 
* **Leakage / imbalance:** strict windowing & stable universe; document deviations. 
