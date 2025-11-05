# PLAN.md — Long‑Term Repo Plan (fjs‑dealias‑portfolio)

**Owner:** Mateo Bodon  
**Advisor:** Prof. Zhou Fan  
**Scope:** MANOVA de‑aliasing overlay for covariance estimation on equity return panels; rigorous calibration vs. MP null/power; fair comparison to shrinkage/factor baselines; portfolio risk evaluation (EW & constrained MV); publishable memo + reproducible code.

---

## 1) Objectives & Success Criteria

**Primary objectives**
1. **Algorithm**: Implement FJS de‑aliasing as an **overlay** (surgical eigenvalue correction only on accepted outliers); support one‑way and nested balanced designs.
2. **Calibration**: Empirically control false positives under the MP null (target **FPR ≤ 2%**) while retaining power vs. planted spikes; freeze default acceptance thresholds (δ, δ_frac, ε, η, energy floor).
3. **Evaluation**: Demonstrate **neutral performance in calm regimes** and **measurable gains in high‑volatility regimes**, over strong baselines (SCM, LW, OAS, Robust SCM, Observed‑factor residuals).
4. **Portfolio**: Report **out‑of‑sample variance MSE**, **VaR/ES coverage**, and **Diebold–Mariano** statistics for EW and **constrained** MV (ridge + box + turnover).
5. **Reproducibility**: One‑command runs, artifact logging (tables, plots, JSON metadata), and memo generation.

**Definition of done (overall)**
- Default acceptance thresholds fixed via null/power ROC.  
- Regime bucket results (Calm/Med/High) show **DM‑significant** gains for de‑aliasing in High; neutral elsewhere.  
- Release notes + memo with figures (edge distributions, acceptance rates, spectra before/after, ΔMSE/coverage tables, DM p‑values).  
- CI green on unit & smoke tests; AWS job templates for heavy sweeps.

---

## 2) Repo Topology (target)

```
.
├── data/
│   ├── wrds/                      # real WRDS extracts (git‑ignored; manifest tracked)
│   ├── processed/                 # panel parquet after cleaning/balancing
│   └── registry.json              # dataset digest (date range, universe, hashes)
├── configs/
│   ├── equity_smoke.yaml          # tiny slice for CI/smoke
│   ├── equity_full.yaml           # full 2018–2024 weekly panel
│   ├── equity_crisis.yaml         # regimes (2020‑Q1/Q2; 2022‑H1)
│   └── acceptance_sweep.yaml      # grid over δ_frac/ε/η/energy_floor
├── src/
│   ├── io/                        # WRDS loaders, parquet utils, manifests
│   ├── design/                    # one‑way/nested balancing, DoW, vol‑state
│   ├── edges/                     # MP edge estimators: SCM/Tyler/Huber
│   ├── dealias/                   # T(λ) root finding, acceptance tests, overlay
│   ├── factor/                    # observed‑factor residualization; POET‑lite
│   ├── portfolio/                 # EW, MV (ridge/box/turnover), VaR/ES
│   ├── eval/                      # MSE, coverage, DM test, summaries
│   └── viz/                       # spectra, edge plots, ROC, heatmaps
├── experiments/
│   ├── synthetic/                 # null/power harnesses
│   └── equity/                    # real‑data runners & sweeps
├── reports/
│   ├── memo/                      # auto‑built memo (md/pdf) with figures
│   └── figures/                   # spectra, ROC, regime tables
├── scripts/                       # thin CLIs; AWS wrappers; WRDS sync
├── tests/                         # unit + smoke; golden figures
├── Makefile
├── AGENTS.md                      # roles & runbooks
├── PLAN.md                        # this file (checked‑in)
├── CODEX_PROMPT.txt               # operational prompt for Codex CLI
└── PROGRESS.md                    # running log of experiments
```

---

## 3) Data & Designs

- **Source**: WRDS (CRSP/Compustat merged) — liquid US equities, **weekly** returns, 2018–2024 (extendable).  
- **Universe**: 100–300 tickers w/ continuous coverage; sector tags retained as attributes.  
- **Designs**:
  - **One‑way**: Week groups (rows) × assets (cols).  
  - **Nested**: Year ⊃ Week‑of‑Year; optionally **vol‑state** buckets (Low/Med/High) derived from realized SPX vol or VIX.
- **Balancing**: Drop incomplete weeks; keep panel rectangular.  
- **Manifests**: `data/registry.json` stores ranges, ticker list, hashes, and feature metadata.

---

## 4) Algorithms & Acceptance (overlay)

- Implement T(λ) root‑finding per FJS; for each candidate outlier band, compute:
  - **t‑vector leakage** off accepted components (threshold ε).  
  - **Energy floor**: minimum explained variance for a direction.  
  - **Stability**: small rotation η‑check; reject unstable roots.  
  - **MP edge**: choose from SCM/Tyler/Huber; maintain low/high edge bands for robustness.  
- **Overlay**: only replace eigenvalues for **accepted directions**; apply standard shrinkage to others (LW/OAS/CC as selected).  
- **Outputs**: `detection_summary.csv` (per‑window edges, accepted idx, thresholds) and `spectra_before_after.png`.

---

## 5) Evaluation Metrics

- **Risk forecast**: OOS variance MSE on EW; OOS MSE for **constrained MV** (ridge = 1e‑4 default; box [0,0.1]; turnover 5–10 bps).  
- **Coverage**: 95% VaR and ES coverage errors.  
- **Statistical tests**: paired **Diebold–Mariano** (loss = squared error) vs LW/OAS/Factor residuals; per‑regime and aggregate.  
- **Diagnostics**: acceptance rate; edge histograms; power/FPR ROC; condition numbers; spectra deltas.

---

## 6) Experiments Matrix (real data)

**Regime buckets** (≥ 50 windows each): Calm, Medium, High.  
**Baselines**: SCM, LW, OAS, Robust SCM, Factor‑residual (observed factors), POET‑lite (optional).  
**Edge modes**: SCM vs Tyler (primary), Huber (ablation).  
**Acceptance sweep**: δ_frac × ε × η × energy_floor — small grid to derive ROC.

Deliverables per run:
- `metrics_summary.csv` (means/medians, CI, DM p‑vals), coverage tables, turnover stats, and figure gallery.

---

## 7) Synthetic Calibration (null/power)

- **Null** (MP bulk only): estimate FPR of acceptance pipeline vs thresholds; target ≤ 2%.
- **Power** (bulk + one spike μ∈{4,6,8}): measure ΔMSE vs LW and acceptance recall.  
- Produce ROC curves and choose default thresholds.  
- Lock defaults in code and note in `CHANGELOG`.

---

## 8) Environments & Compute

- **Local dev**: conda env `dealias-env`; deterministic seeds; lightweight smoke tests.  
- **AWS** (new):
  - **EC2 template**: m6i.2xlarge (or GPU if needed later); EBS 200GB; IAM role with S3 read/write.  
  - **S3 buckets**: `s3://fjs-dealias/artifacts/` for CSV/figures; `s3://fjs-dealias/datasets/` for WRDS extracts.  
  - **Runner script**: `scripts/aws_run.sh` to rsync repo, run `make`, stream logs, and sync artifacts back.

**Secrets**: store WRDS creds via `~/.wrds.cfg` on EC2; never commit. Use AWS Parameter Store for any tokens.

---

## 9) Makefile Targets (proposed)

- `make env` — create/lock conda; install repo.  
- `make data:sync_wrds` — refresh WRDS → `data/wrds/`; update `registry.json`.  
- `make build:processed` — build balanced parquet panels.  
- `make run:equity_smoke` — tiny slice end‑to‑end (CI).  
- `make run:equity_full` — full weekly panel with defaults.  
- `make sweep:acceptance` — δ_frac/ε/η grid; write ROC figures.  
- `make run:regimes` — Calm/Med/High buckets; produce tables + memo.  
- `make memo` — build `reports/memo/memo.pdf`.  
- `make test` — unit + smoke; golden figures.  
- `make aws:<target>` — wrap any of the above on EC2 (provisions if missing).  
- `make clean` — purge intermediates (keep dataset registry).

---

## 10) CI/CD

- **GitHub Actions**:
  - Lint + unit + smoke on PR.  
  - Artifact upload (metrics/figures from smoke).  
  - Optional manual dispatch: trigger small acceptance sweep; attach ROC PNGs.

---

## 11) Documentation & Logs

- `PROGRESS.md`: structured log (date, SHA, config, dataset digest, key deltas, links to artifacts).  
- `reports/memo/` renders markdown → PDF with embedded tables/plots.  
- Every run writes a `run.json` metadata blob (config, seeds, env, git SHA, timing).

---

## 12) Governance

- **Conventional commits**; CHANGELOG per release.  
- Issues labeled: algo, data, infra, eval, docs.  
- PR template: problem, approach, tests, artifacts, risk, checklist.

---

## 13) Risks & Mitigations

- **MV instability** → ridge/box/turnover defaults; sanity checks on Σ condition number.  
- **Over‑acceptance** → null FPR calibration; robust edge modes; stability checks.  
- **Data drift** → dataset registry; lock universes by manifest.  
- **Compute cost** → regime subsampling; AWS spot with checkpointable sweeps.

---

## 14) Timeline (6 weeks)

**W1**: Data registry; balanced panels; smoke/full config; unit tests for design & edges.  
**W2**: Null/power harness + ROC; freeze defaults; document acceptance.  
**W3**: Portfolio stack solid (MV constraints; turnover; coverage); DM testing in pipeline.  
**W4**: Regime buckets; primary results; memo v1.  
**W5**: Ablations (factor residuals; Tyler vs SCM edges); robustness appendix.  
**W6**: Polish: CI, docs, release v0.1; advisor memo.



