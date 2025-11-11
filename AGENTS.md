# AGENTS.md — vNEXT (fjs‑dealias‑portfolio)

This file instructs coding agents (e.g., Codex CLI) how to work on this repo.

## Mission
Implement and evaluate a **de‑aliasing overlay** for covariance estimation on equity panels. Keep acceptance well‑calibrated, compare against strong baselines, and generate advisor‑ready reports.

## Ground rules
- Never commit secrets. WRDS credentials live in `~/.wrds.cfg` on machines; S3 keys via AWS profile/role. Do not write credentials into code or logs.
- Prefer **overlay** to full spectrum replacement; non‑accepted directions default to a shrinker (RIE/LW/OAS/CC).
- **Real data for tests**: Smoke/integration can use the bundled sample, but all RC/ablation runs must read WRDS CSV/Parquet tracked in `data/registry.json`.
- Determinism first: pin thread caps to 1 for heavy math and record seeds; throughput mode only for exploratory work.
- Always write artifacts: CSV/PNGs/JSON, plus `run_manifest.json` (git SHA, dataset digest, config, seeds, instance, exec mode).

## Setup commands
- Create env and install: `make setup`
- Quick checks: `make fmt && make lint && make test-fast`
- Full tests: `make test` (slow tests are opt‑in via markers)
- Smoke run (local): `make run:equity_smoke`
- RC batch (local): `make rc-lite`
- AWS provision: `scripts/aws_provision.sh` (export `INSTANCE_TYPE`, `INSTANCE_FAMILY`, `INSTANCE_SIZE` when needed)
- AWS RC: `EXEC_MODE=deterministic make aws:rc-lite`
- Calibration sweep (AWS): `EXEC_MODE=deterministic make aws:calibrate-thresholds HARNESS_TRIALS=600`

## Paths & data contracts
- Real panels live under `data/wrds/*.parquet` or `data/returns_daily.csv` (long: `date,ticker,ret`); update `data/registry.json` via `tools/update_registry.py` after ingest.
- Factor files (for prewhitening) are CSV with columns = factors (e.g., `MKT,SMB,HML,RMW,CMA,UMD`). Store path in configs; never commit WRDS raw extracts.

## Common tasks
- **Add prewhitening to daily/RC**: plumb `--prewhiten` and `--factor-csv` into runners; persist R² and factor metadata; update memo tables to include factor baselines.
- **Raise nested coverage**: relax `delta_frac_min`, confirm replicate balancing, ensure `q_max` allows up to 2 substitutions; re‑run nested smoke.
- **Re‑calibrate acceptance**: run synthetic ROC on AWS; refresh `calibration_defaults.json`; rebuild smoke/RC with the new defaults.
- **RC (WRDS)**: DoW + vol‑state, top‑N assets, prewhitening on/off; build gallery, memo, brief; attach `kill_criteria.json` and limitations to summary.

## Coding conventions
- Python 3.11+, `black` + `ruff` + `mypy`. Prefer functional helpers for deterministic math. Avoid global state; pass seeds explicitly.
- Commit style (Conventional Commits):  
  - `feat(dealias): ...`, `fix(eval): ...`, `chore(ci): ...`, `docs(memo): ...`
- Branching: `feat/<ticket>` for features, `docs/<ticket>` for documentation. PRs include: problem, approach, tests, artifacts, risks, checklist.

## Testing
- Unit: math utils, MP edges, gating decisions (deterministic).  
- Integration: smoke slice end‑to‑end (small universe).  
- Slow: synthetic ROC + ablations (tagged).  
- Always run `make test` before committing multi‑file changes. Prefer `pytest -q` locally; CI runs unit + smoke.

## Artifacts & telemetry
- Per run: `metrics.csv`, `dm.csv`, `risk.csv`, `diagnostics*.csv`, plots, and `run_manifest.json`.  
- RC: gallery under `figures/rc/<tag>/`, memo `reports/memo.md`, brief `reports/brief.md`, plus kill‑criteria and limitations JSON.

## AWS discipline
- Use `tools/run_monitor.py` wrappers; keep `MONITOR_INTERVAL` reasonable (e.g., 5–10s).  
- Prefer `EXEC_MODE=deterministic` for calibration/RC; set BLAS thread caps (OMP/MKL/OPENBLAS/NUMEXPR=1).  
- Upload artifacts to `s3://<bucket>/reports/<tag>/` via make targets; avoid ad‑hoc copies.

## Safety & approvals (Codex)
- Default to **danger‑full‑access**, **network enabled**, **approval_policy=never** only when you are in a trusted, ephemeral environment (local dev box or dedicated EC2) and Git is clean with a branch. Otherwise use `on‑failure`.
- Never run destructive commands (`rm -rf`, `git reset --hard`, force pushes) without existing backups and a clear ticket scope.
