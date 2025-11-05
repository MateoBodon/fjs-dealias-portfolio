# AGENTS.md — Roles & Runbooks

## Agent roster

- **Orchestrator**: splits work, sequences tasks, ensures DoD is met.  
- **DataEngineer**: WRDS ingest, balancing, manifests, S3 sync, reproducibility.  
- **Statistician**: MP edges, de‑alias acceptance, null/power calibration, defaults.  
- **PortfolioAnalyst**: EW/MV pipelines, coverage, DM tests, constraints/turnover.  
- **Evaluator**: experiment matrix, regime bucketing, artifact dashboards, memo.  
- **ReleaseManager**: CI, Makefile, tagging, CHANGELOG, PROGRESS.md.

## Shared constraints

- Never commit secrets; use manifests & hashes; attach `run.json` to each artifact set.  
- Prefer overlay; avoid full spectrum replacement; justify threshold choices with ROC.

## Standard operating procedure (per task)

1) Create an issue with scope, inputs, expected outputs, acceptance tests.  
2) Branch `feat/<ticket>`; implement with tests.  
3) Run `make test` + relevant run target; capture artifacts (CSV/PNG/JSON).  
4) Update `PROGRESS.md` with SHA, config, dataset digest, key tables/figures.  
5) PR with artifacts attached; merge on green CI.

## Commit conventions

`feat(dealias): overlay acceptance with ε/η gates`  
`fix(portfolio): add ridge & box constraints to MV`  
`docs(memo): add ROC figure and default thresholds`  
`chore(ci): upload smoke artifacts`  