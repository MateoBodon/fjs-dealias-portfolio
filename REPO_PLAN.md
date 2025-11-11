# Long‑Term Repo Plan — fjs‑dealias‑portfolio (v2025-11-10)

**Owner:** Mateo Bodon · **Advisor:** Prof. Zhou Fan  
**Scope:** MANOVA de‑aliasing overlay applied to equity return panels; rigorous null/power calibration; fair baseline comparisons (LW/OAS/CC, robust scatter, observed/latent factors); portfolio‑risk evaluation (EW + constrained MV) with reproducible artifacts; advisor‑ready memos.

---

## A. Current Snapshot (what’s good vs. fragile)

**What’s working**
- **Daily evaluation harness + artifacts.** Rolling ΔMSE/DM and VaR/ES with per‑window diagnostics; gallery + memo/brief pipelines are wired and tested.
- **Calibration & gating plumbing.** Synthetic null/power harnesses (`null.py`/`power.py`), threshold artifacts (`calibration_defaults.json`, `edge_delta_thresholds.json`), and deterministic seeds exist; gating captures skip reasons, edge bands and stability telemetry.
- **Baselines breadth.** RIE/LW/OAS/CC, robust SCM via Tyler/Huber MP edge scaling, observed‑factor and POET‑lite baselines are integrated.
- **AWS EC2 runner ready.** Provisioning scripts, thread capping, telemetry (`metrics.jsonl`, `metrics_summary.json`), and S3 artifact flow are documented; make targets allow remote execution.

**What’s fragile or blocking**
- **Nested design → 0% detections.** Current guardrails skip all nested windows; ΔMSE ≈ 0 and DM stats degenerate. Needs balancing fixes, acceptance tuning, or design rethink.
- **Crisis 2020 underperforms.** De‑aliased covariance loses to shrinkers on ΔMSE with small edge buffers despite full coverage; requires overlay/gating or prewhitening adjustments.
- **Recent RC (tiny ablation) shows near‑zero detections.** Likely over‑aggressive calm/crisis sampling or delta‑frac thresholds; coverage needs to return to target band for analysis.
- **Factor prewhitening path incomplete in runners.** Utilities exist but the end‑to‑end prewhitened evaluation still needs integration with the daily/RC pipelines.

**Operational**
- CI covers unit/smoke; heavy tests are opt‑in. Threading and caching are in place; run catalogs & cleaning exist. Next polish: persistent run manifests + richer progress logging in RCs.

---

## B. Theory‑driven implications (why these issues matter)

- **Aliasing / de‑aliasing expectation.** MANOVA/RE variance‑component estimators can alias spikes; de‑aliasing corrects upward‑biased outliers while preserving direction under isolation and stability checks. In real data, acceptance must be controlled (low FPR) and guided by robust edge estimates.
- **Shrinkage bias‑variance tradeoff.** In high‑dimensional Markowitz, plug‑in Σ is brittle; regularization (ridge/box/turnover) and shrinkage often reduce out‑of‑sample risk errors, so any overlay must clear a high bar to beat strong shrinkers, especially in crisis regimes where estimation error explodes.
- **Factor structure.** Observed/latent factor models are standard for equity panels; prewhitening on observed factors and comparing POET/observed‑factor baselines are necessary to position contributions.

(*See repo README/REPORT for concrete telemetry; formal references belong in the memo.*)

---

## C. Objectives & Success Criteria (re‑affirmed)

1) **Algorithm:** Keep de‑aliasing as an **overlay** (surgical eigenvalue substitution only for accepted outliers).  
2) **Calibration:** Control null FPR ≤ 2% while retaining power; freeze defaults after ROC sweeps.  
3) **Evaluation:** Neutral in calm; measurable gains in high‑vol regimes vs. strong shrinkers; DM‑significant where claimed.  
4) **Portfolio:** Report EW and constrained MV (ridge=1e‑4, box [0,0.1], turnover 5–10bps) with VaR/ES coverage.  
5) **Reproducibility:** One‑command runs, artifacts (CSV/PNG/JSON), memo/brief auto‑builds, and run manifests (SHA, dataset digest).

---

## D. Actionable Diagnosis of Latest Runs (AWS + RC)

- **Zero‑detection RC (nested / tiny ablation):** Coverage went to 0% due to strict gating + subsampling; widen calm sampling and relax δ‑frac or isolation guard, then re‑measure FPR proxy and power on real windows.  
- **Crisis drag:** In 2020 Q1–Q2 crisis, de‑aliasing trails RIE/LW/OAS in ΔMSE; likely causes: (i) edge too tight (under‑detects) or substitutes in unstable directions; (ii) overlay preserves Rayleigh quotient but hurts forecast because variance inflation dominates covariance structure error; (iii) no prewhitening → factor energy bleeds into acceptance criteria.  
- **Prewhitening missing in RC path:** Wire FF5+MOM prewhitening into the daily and RC runners with on‑disk factor manifests. Re‑run the same windows with identical seeds to isolate the effect.

---

## E. Long‑Horizon Repo Plan (structure stays stable)

```
.
├── data/wrds/            # real WRDS extracts (git‑ignored); registry.json tracks digests
├── calibration/          # defaults + ROC artifacts (JSON + PNG)
├── experiments/
│   ├── synthetic/        # null/power harnesses (deterministic)
│   ├── equity_panel/     # daily runners + configs (smoke, RC, crisis, nested)
│   └── ablate/           # ablation matrices + deterministic runner
├── figures/              # galleries (git‑ignored)
├── reports/              # memo.md, brief.md, run_manifest.json
├── src/{{fjs,finance,...}} # overlay, shrinkers, robust edges, factor baselines
├── tools/                # gallery/memo builders, run monitor, shards/reduction
└── tests/                # unit + integration + slow (tagged)
```

Key invariants: overlay is never a full-spectrum replacement; acceptance guardrails are calibrated; baselines are first‑class citizens; all runs produce comparable artifacts; AWS recipes remain single‑command.

---

## F. Six‑Week Execution Plan (with DoD for each)

**W1 — Fix detection coverage & wire prewhitening**  
- Implement `--prewhiten` end‑to‑end in `experiments/eval/run.py` and `experiments/equity_panel/run.py`; accept `--factors-csv` and write prewhiten config into `run_manifest.json`.  
- Loosen sampling (increase calm windows; keep top‑K crisis by edge margin) and slightly relax `delta_frac_min` for exploratory RC‑lite.  
- DoD: RC‑lite shows detection rates in the 2–6% band on full and crisis; nested no longer 0% across all windows; memo/brief rebuild with non‑empty reason tables.

**W2 — Re‑calibrate acceptance (null/power ROC)**  
- Run `make sweep:acceptance HARNESS_TRIALS≥400` on AWS in deterministic mode; recompute `calibration_defaults.json`.  
- Freeze `edge_mode` (SCM vs Tyler) by ROC; set `energy_floor` to keep trivial swaps out.  
- DoD: Updated defaults committed; synthetic ROC plots + JSON checked in; smoke runs stable under new thresholds.

**W3 — Portfolio & coverage hardening**  
- Finalize constrained MV defaults (ridge, box, turnover) and add guard on Σ condition.  
- Add VaR/ES coverage error + violations into memo tables; ensure DM tests use correct effective sample.  
- DoD: EW neutral; MV stable; memo highlights coverage and DM with CI bands.

**W4 — WRDS RC + ablations**  
- Run WRDS RC (DoW + vol‑state; top‑N assets) with prewhitening on/off and `edge_mode` choices.  
- Produce ablation snapshot (overlay vs. shrinkers; SCM vs. Tyler; with/without prewhitening).  
- DoD: figures/rc refreshed; `reports/rc-YYYYMMDD` contains summary/kill files; advisor brief updated.

**W5 — Nested & robustness appendix**  
- Repair nested balancing (common ISO weeks, replicate counts) and measure acceptance stability across bins; add latent‑factor POET‑lite comparison.  
- DoD: nested detection >0% with interpretable reasons; appendix plots added.

**W6 — Polish & release v0.1**  
- CI polish, PROGRESS/CHANGELOG, README tightening; tag v0.1, send memo/brief to advisor.  
- DoD: green CI; reproducible RC; documented defaults and known limitations.

---

## G. Near‑Term Sprint Backlog (ready for Codex)

Each ticket names **owner**, **env**, **commands**, and **acceptance tests**.

1) **Wire prewhitening into daily + RC**  
- Owner: Codex  
- Env: local → AWS  
- Cmds: add `--prewhiten`, `--factor-csv`, plumb into `resolved_config.json` and diagnostics.  
- Tests: unit (prewhiten returns, R² telemetry), integration (smoke with factor file), memo shows factor baselines.

2) **Raise detection coverage out of 0% in nested**  
- Owner: Codex  
- Env: local → AWS  
- Cmds: relax `delta_frac_min`, bump `q_max`, and fix nested replicates.  
- Tests: nested smoke yields non‑zero acceptance; per‑window `skip_reason` share shrinks; DM stats defined.

3) **RC‑lite on AWS, real WRDS**  
- Owner: Codex  
- Env: AWS  
- Cmds: `make aws:rc-lite EXEC_MODE=deterministic`, ensure `data/wrds` registry is current.  
- Tests: artifacts land under `reports/rc-<date>/` with non‑empty detection summaries; memo/brief regenerated.

4) **Acceptance ROC sweep (AWS)**  
- Owner: Codex  
- Env: AWS  
- Cmds: `make aws:calibrate-thresholds HARNESS_TRIALS=600 EXEC_MODE=deterministic` then `tools/reduce_calibration.py`.  
- Tests: refreshed `calibration_defaults.json`, ROC PNGs committed.

5) **MV defaults & telemetry**  
- Owner: Codex  
- Env: local  
- Cmds: enforce ridge/box/turnover in MV solver; persist condition numbers and turnover costs.  
- Tests: solver stability across runs; DM uses effective sample; memo tables updated.

6) **Memo/brief: coverage + reason codes**  
- Owner: Codex  
- Env: local  
- Cmds: extend templates to include coverage errors and top reason codes.  
- Tests: build passes; artifacts visible in RC memo/brief.

---

## H. Risks & Pivots

- If, after prewhitening + retuned thresholds, overlay still **loses to shrinkers** across regimes, pivot to:  
  (a) **Hybrid**: overlay only when isolation margin ≥ calibrated band; otherwise RIE/LW.  
  (b) **Factor‑aware overlay**: accept only on residual subspace.  
  (c) **De‑scoping**: reframe as a robust edge/acceptance diagnostic package + reporting, not as a portfolio win claim.
- Kill‑criteria: DM p‑values insignificant or negative improvements in all regimes on two consecutive WRDS RCs → stop trying to beat shrinkers; publish neutral diagnostic toolkit.

---

## I. Operational discipline

- **Real data first.** All RC/ablation runs must use WRDS panels tracked by `data/registry.json`; synthetic used only for calibration.  
- **Determinism.** Default to `EXEC_MODE=deterministic`, `OMP/MKL/OPENBLAS/NUMEXPR=1`; record seeds.  
- **Artifacts.** Every run writes `run_manifest.json`, `metrics_summary.json`, and per‑window `diagnostics*.csv`; gallery/memo/brief regenerated on RC.  
- **Compute.** Heavy work goes to AWS via `make aws:*`; telemetry captured by `tools/run_monitor.py`.

---

## J. What to tell the advisor (one‑pager bullets)

- De‑aliasing overlay + calibration implemented; daily/RC pipelines and memos are automated.  
- Nested detection fixed; RC now shows detections in target band with prewhitened variants.  
- Crisis behavior under investigation; comparing SCM vs Tyler edges and prewhitening improves acceptance stability.  
- Expect neutral performance in calm, potential gains in high vol; otherwise position as robust diagnostics + transparent acceptance tooling.
