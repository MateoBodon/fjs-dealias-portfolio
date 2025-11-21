# PLAN.md — Long‑Term Repo Plan (fjs‑dealias‑portfolio, v2 — Nov 2025)

Owner: Mateo Bodon  
Advisor: Prof. Zhou Fan  

Scope: FJS-style MANOVA de-aliasing overlay on equity return panels, calibrated against MP null/power and compared to shrinkage/factor baselines, with portfolio-level risk metrics and reproducible RC-style drops.

---

## 0. Current Status Snapshot (Nov 2025)

What’s already in good shape (don’t break this):

- **Core runners are real and non-trivial.**
  - `experiments/equity_panel/run.py` runs one-way and nested weekly designs on CRSP-style WRDS returns with prewhitening and gating flags.:contentReference[oaicite:0]{index=0}  
  - `experiments/eval/run.py` gives daily overlay diagnostics + VaR/ES metrics by regime (full/calm/crisis).:contentReference[oaicite:1]{index=1}  

- **Calibration + gating infrastructure exists.**
  - Synthetic null/power harness (`experiments/synthetic/null.py`, `power.py`, `calibrate_thresholds.py`) plus `make sweep:acceptance` and `make calibrate-thresholds` write ROC tables and `calibration_defaults.json` / `calibration/edge_delta_thresholds.json`.:contentReference[oaicite:2]{index=2}  
  - Gating is wired via `calibration/edge_delta_thresholds.json` and per-window `detection_summary.csv` with skip reasons, edge margins, and alignment stats.:contentReference[oaicite:3]{index=3}  

- **RC-style equity runs with real metrics exist.**
  - `make rc` / `make rc-lite` orchestrate smoke, nested, and crisis batches; generate gallery + memo + brief under `figures/rc/...` and `reports/rc-YYYYMMDD/`.:contentReference[oaicite:4]{index=4}  
  - Latest RC telemetry (4 Nov 2025) shows:
    - DoW design (RIE, FF5+MOM): detection ~3–5%, substitution ~5%, small but nonzero ΔMSE vs shrinkers, reasonably controlled “FPR surrogate”.  
    - Vol-state design (OAS, no prewhiten): detection ~3–4%, acceptable substitution; VaR95 coverage errors ~±1%.:contentReference[oaicite:5]{index=5}  

- **Prewhitening & factor baselines are integrated.**
  - Observed-factor and POET-lite estimators are first-class options (`--estimator factor_obs`, `poet`), with factor CSVs and prewhitening flags, all recorded into rolling results + memos.:contentReference[oaicite:6]{index=6}  

- **Testing and reporting are not toys.**
  - `make test-fast`, `make test-integration`, `make test-slow`, `make test` drive a sensible pytest marker split.  
  - Gallery and memo/brief tooling (`tools/build_gallery.py`, `build_memo.py`, `build_brief.py`) assemble advisor-ready markdown with DM stats, ΔMSE, gating diagnostics, etc.:contentReference[oaicite:7]{index=7}  

Key open issues (based on the README + RC notes):

- Nested design currently has **0% accepted detections** in some slices because guardrails are too strict (memo badges “no accepted detections; check guardrails”).:contentReference[oaicite:8]{index=8}  
- 2020 crisis runs show de-aliased ≫ shrinkage MSE even when detections are plentiful — overlay is too aggressive for that regime.:contentReference[oaicite:9]{index=9}  
- Ablation grid (`config.ablation.smoke.yaml`) is timing out before finishing.:contentReference[oaicite:10]{index=10}  
- Infra/docs assume AWS; you now also have a Hetzner box and want to lean on it for all heavy calibrations.

This plan is about closing those gaps and getting to a “Prof‑Fan‑ready” state.

---

## 1. High‑Level Objectives

1. **Algorithmic overlay is mature.**
   - De-aliasing overlay is well-behaved on synthetic null/power and on WRDS equities (DoW + nested + vol-state).
   - MP edge modes (SCM / Tyler / Huber) and gating thresholds are calibrated and frozen.

2. **Risk results are honestly reported.**
   - For EW and constrained MV portfolios you have:
     - ΔMSE vs strong shrinkage/factor baselines.
     - VaR/ES coverage and ES error.
     - Diebold–Mariano and sign tests, especially on the “flip set” (windows actually affected by overlay).:contentReference[oaicite:11]{index=11}  

3. **Release Candidates (RCs) are reproducible.**
   - A single `make rc` or `make rc-lite-sanity` on a machine with WRDS exports reproduces the same `reports/rc-YYYYMMDD` drop (memo + brief + figures).:contentReference[oaicite:12]{index=12}  

4. **Repo is Codex‑friendly.**
   - `AGENTS.md`, `PLAN.md`, and a clean `~/.codex/config.toml` profile guide Codex to:
     - use WRDS data when evaluating,
     - run the right tests,
     - document work in `PROGRESS.md` and RC memos,
     - commit/push to dedicated branches safely.

---

## 2. Repo Structure & Conventions (Target)

You already have most of this; this just formalizes it so Codex and humans behave consistently.:contentReference[oaicite:13]{index=13}  

- `data/`
  - `returns_daily.csv` — canonical WRDS-aligned daily panel (date,ticker,ret).
  - `factors/ff5mom_daily.csv` — FF5+MOM factor panel (date,<factors>).
  - `registry.json` — dataset digests (source, date span, SHA256, row count).  
  - `wrds/` — raw WRDS exports (ignored by git).

- `experiments/`
  - `equity_panel/` — weekly designs (oneway, nested, vol-state) + configs:
    - `config.smoke.yaml`, `config.crisis.2020.yaml`, `config.crisis.2022.yaml`, `config.ablation.smoke.yaml`, `config.gallery.yaml`, `config.rc.yaml`, etc.:contentReference[oaicite:14]{index=14}  
  - `eval/` — daily overlay runner.
  - `etf_panel/` — ETF demo wrapper (country/sector).:contentReference[oaicite:15]{index=15}  
  - `synthetic/` — null/power calibration and threshold sweeps.

- `src/`
  - `finance/` — covariance + shrinkage + robust SCM.
  - `fjs/` — de-aliasing transforms and acceptance logic.
  - `evaluation/` — metrics, DM stats, regime splits.
  - `report/` — table/plot assembly.
  - `meta/` — misc helpers (caching, registry handling).

- `tools/`
  - Cleaning: `clean_outputs.py`.  
  - Run summarization: `summarize_run.py`, `prewhiten_effect.py`, `list_runs.py`.:contentReference[oaicite:16]{index=16}  
  - Calibration utilities: `shard_grid.py`, `reduce_calibration.py`, `update_registry.py`.

- `reports/`
  - `rc-YYYYMMDD/` — per-RC manifests, metrics, diagnostics, memos, briefs.
  - `synthetic/` — null/power ROC, calibration runs.
  - `templates/` — Jinja templates for memos/briefs.

- `figures/`
  - `rc/YYYYMMDD/**` — RC plots.
  - `summary/**` — edge histograms, isolation share, stability scatter.:contentReference[oaicite:17]{index=17}  

- Project hygiene:
  - `tests/`, `.github/workflows/smoke.yml`, `AGENTS.md`, `PLAN.md` (this), `PROGRESS.md`, `RUNBOOK.md`, `ROADMAP.md` as needed.

---

## 3. Phased Plan

### Phase 1 — Codex + Infra Hygiene (immediate, 1–2 weeks)

**Goal:** Make the repo “AI-first”: Codex can drop in, understand the context, run tests, and ship small improvements safely on both your laptop and the Hetzner box.

Tasks:

1. **AGENTS.md overhaul (see template below).**
   - Spell out:
     - Setup (`pip install -e .[dev]`, `make test-fast`).  
     - Data expectations: WRDS CSVs live under `data/`, real WRDS for experiments, synthetic only for smoke.:contentReference[oaicite:18]{index=18}  
     - Test commands: `make test-fast`, `make test`, `make rc-lite-sanity`.  
     - Run commands: `make rc`, `make sweep:acceptance`, `python experiments/eval/run.py ...`.:contentReference[oaicite:19]{index=19}  
     - Git/branching: `codex/<short-task>` branches, conventional commit prefixes.

2. **Codex config + profiles.**
   - Create `~/.codex/config.toml` with:
     - Default model `gpt-5.1-codex-max`.:contentReference[oaicite:20]{index=20}  
     - `profiles.fjs-local` (moderate approvals, workspace-write).  
     - `profiles.fjs-hetzner` (approval_policy `never`, `danger-full-access`, web search enabled).

3. **Hetzner integration.**
   - Mirror the repo + WRDS data onto Hetzner (you already did basic tests).
   - Extend `docs/CLOUD.md` (or new `docs/HPC.md`) to describe:
     - How to ssh in.  
     - Paths for WRDS data (`/mnt/wrds` etc).  
     - Recommended commands for heavy jobs:
       - `EXEC_MODE=throughput make calibrate-thresholds`  
       - `EXEC_MODE=throughput make rc` / `rc-lite`.:contentReference[oaicite:21]{index=21}  

4. **Basic smoke CI sanity.**
   - Ensure `make test-fast` and a tiny slice of equity runs (`config.smoke.yaml` with `assets-top 80`, `stride-windows 4`) are stable locally and on Hetzner.:contentReference[oaicite:22]{index=22}  
   - Codex should always run `make test-fast` before committing.

**Definition of done (Phase 1):**

- AGENTS.md exists and is Codex-friendly.  
- `config.toml` profiles work on both laptop and Hetzner.  
- One Codex-driven PR/branch that:
  - runs tests,  
  - updates PROGRESS.md,  
  - tweaks something small (docs or code) end-to-end.

---

### Phase 2 — Nested Detection & Guardrails (short‑term, 2–3 weeks)

**Goal:** Stop the nested design from being dead (0% accepted detections) while keeping null FPR under control.

Tasks:

1. **Reproduce the bad nested run.**
   - Use the exact config that produced “0/24 windows accepted, everything skipped by guardrails” (nested smoke 2022-01→2023-12).:contentReference[oaicite:23]{index=23}  
   - Codex task: locate the config + output directory, confirm diagnostics (skip reasons, gating section in `summary.json`, nested skip details).

2. **Instrument nested diagnostics.**
   - Add more fields to `detection_summary.csv` and `summary.json` for nested runs if needed:
     - `nested_years_kept`, `nested_common_weeks`, per-year sample size, etc. (some of this already exists; tighten it).:contentReference[oaicite:24]{index=24}  

3. **Guardrail tuning loop.**
   - Using synthetic harness + nested null design:
     - Sweep:
       - minimum isolation conditions,
       - alignment thresholds,
       - `q_max` per window,
       - nested-specific stability tolerances.:contentReference[oaicite:25]{index=25}  
     - Record nested FPR and power for these configurations.  
   - On WRDS nested runs:
     - Aim for detection coverage in the 2–6% band, not 0%.  
     - Ensure skip reasons are not dominated by “no_isolated_spike” + hyper-strict stability.

4. **Codex-friendly automation.**
   - Add a Make target, e.g., `make nested-sanity`, that:
     - runs the nested smoke config,  
     - writes a compact nested-only `metrics_nested.csv`,  
     - updates a `reports/rc-YYYYMMDD/nested_summary.md`.

**Definition of done (Phase 2):**

- Nested smoke and at least one nested crisis slice show nonzero coverage.  
- Null FPR still ≤ target (e.g. 2%) under synthetic nested null.  
- Nested telemetry summarized in memo/brief without the “no accepted detections” badge.

---

### Phase 3 — Crisis Tuning & Ablations (medium‑term, 3–5 weeks)

**Goal:** Make crisis slices not look embarrassing (overlay should be neutral or modestly helpful, not strictly worse than shrinkage).

Tasks:

1. **Deep dive on 2020 crisis.**
   - Use `experiments/equity_panel/outputs_crisis_2020/...` for de-aliased vs shrinkers.:contentReference[oaicite:26]{index=26}  
   - Codex should compile:
     - ΔMSE vs LW/OAS/Tyler for EW and MV.  
     - DM p-values and sign tests on the flip set (`dm_flip_only.csv`).:contentReference[oaicite:27]{index=27}  
     - Edge margin distributions.  

2. **Overlay throttle strategies.**
   - Experiment with:
     - Lowering `delta_frac_min` in crisis regimes.  
     - Increasing required edge margins in crisis only.  
     - Limiting `q_max` to 1 under crisis gating.  
   - Evaluate the tradeoff:
     - ΔMSE vs shrinkers.  
     - VaR/ES coverage and violation rates.:contentReference[oaicite:28]{index=28}  

3. **Finish ablation grid.**
   - Either:
     - shrink `config.ablation.smoke.yaml` (fewer dimensions) or  
     - raise timeouts and run it on Hetzner with throughput mode.:contentReference[oaicite:29]{index=29}  
   - Output `ablation_summary.csv` and ensure `make gallery` plots `ablation_heatmap.png`.:contentReference[oaicite:30]{index=30}  

4. **Codex automation for crisis RCs.**
   - Make `make rc-crisis` which:
     - runs the crisis configs (2020, 2022).  
     - rebuilds a crisis-focused gallery and memo/brief subset.  
   - Add an “Ablations” section to the memo template that auto-embeds the ablation heatmap when present.:contentReference[oaicite:31]{index=31}  

**Definition of done (Phase 3):**

- Crisis RC memo: overlay no longer obviously worse; ideally neutral in calm and non-catastrophic in crisis.  
- Ablation heatmap exists and is referenced in the memo.  
- Codex can run `make rc` on Hetzner and produce a fresh RC drop in a single session.

---

### Phase 4 — Factor/FJS Positioning vs Shrinkage (medium‑term, 4–6+ weeks)

**Goal:** Produce a story coherent with Fan–Jiang–Sun + Markowitz + factor literature: where does the FJS overlay sit relative to Ledoit–Wolf, OAS, robust SCM, observed-factor, and POET-like estimators?  

Tasks:

1. **Factor baselines sanity.**
   - Confirm that:
     - Observed-factor covariance (FF5+MOM) and POET-lite are stable and recorded in `metrics_summary.csv` and memos.:contentReference[oaicite:33]{index=33}  
     - Prewhitening diagnostics via `tools/prewhiten_effect.py` are understandable and used for at least one vol-state comparison (off vs FF5+MOM).:contentReference[oaicite:34]{index=34}  

2. **Under-the-hood alignment checks.**
   - On synthetic and equity panels:
     - Compare accepted spike directions to factor loadings.  
     - Check whether overlay is “rediscovering” factor structure or correcting residual spikes.

3. **Portfolio-level Markowitz tests.**
   - On the daily evaluation harness (`experiments/eval/run.py`):​:contentReference[oaicite:35]{index=35}  
     - EW and constrained MV (ridge + box + turnover).  
     - Compare overlay vs each shrinker/factor baseline using DM tests, VaR/ES coverage, and ES error.  

4. **Summarise for Prof. Fan.**
   - Generate a stable RC drop + memo focusing on:
     - Null/power calibration results.  
     - DoW/nested/vol-state performance vs shrinkage and factor baselines.  
     - Crisis vs calm behavior.  
     - When de-aliased overlay adds value, when it’s neutral, and when shrinkage wins.

**Definition of done (Phase 4):**

- A single RC “release” directory with memo + brief that you’re comfortable emailing to your advisor as a draft “results section”.

---

### Phase 5 — “Paperization” and Hardening (long‑term)

This is the “if we decide to turn this into a real paper” phase:

- Lock default calibrations and clearly version them (e.g., `calibration/edge_delta_thresholds.v1.json`).  
- Add more rigorous portfolio constraints, robustness checks, small variations in design (e.g. alternative groupings beyond DoW/vol-state).  
- Expand tests for numerical robustness (condition numbers, near-singular cases).  
- Formalize environment + data ingestion so the whole thing can be rebuilt from WRDS creds and code alone.

---

## 4. Operational Expectations for Codex

Codex should obey these rules by default:

1. **Always read `AGENTS.md` + `PLAN.md` before doing anything non-trivial.**
2. **Never commit WRDS raw data or secrets.**
3. **For every non-trivial change:**
   - Run `make test-fast` (at least).  
   - If you touched runners/evaluation: run `make rc-lite-sanity`.  
   - Update `PROGRESS.md` with date, commit, configs used, and key metrics.  
   - Commit on a `codex/<short-task>` branch with a clear message.  
4. **For heavy calibrations or RCs,** prefer running on the Hetzner profile (`codex --profile fjs-hetzner` on the server).

