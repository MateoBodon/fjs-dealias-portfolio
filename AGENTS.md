# AGENTS.md — fjs-dealias-portfolio
_Last updated: 2025-11-02_

This repo uses **Codex** (GPT‑5‑Codex) and other coding agents. This file is a **README for agents**.

## Mission
Implement and evaluate a **de‑aliasing overlay** for portfolio risk. Daily estimation with replicated groups; overlay only when detections are strong; compare to shrinkage/factors; produce reproducible RC artifacts.

## Environment
- Python ≥ 3.11
- Use a virtualenv `.venv`
- Prefer `pip` (or `uv`) and editable installs

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]    # add [dev] extras if missing
```
If Makefile exists:
```bash
make setup
```

## Commands (copy‑pasteable)
- **Run tests**: `pytest -q`
- **Lint**: `ruff check .`
- **Typecheck (optional)**: `mypy src`
- **Calibrate thresholds**: `python scripts/calibrate_thresholds.py --p 200 --n 252 --mu 4,6,8 --out reports/calibration/`
- **Evaluate (rolling)**: `python experiments/eval/run.py --regime crisis --out reports/rc-YYYYMMDD/`
- **Build memo & gallery**: `make rc && make gallery`

## Do / Don’t
### Do
- Use **robust edge** (Tyler/Huber) by default.
- Treat de‑aliasing as an **overlay**: replace eigenvalues only for detected directions, preserve eigenvectors, shrink the rest.
- Tune thresholds to **FPR ≤ 1–2%** at matched p/n via synthetic calibration.
- Log **detection rate**, **isolation share**, **edge margin**, **direction stability** per window.
- Keep changes **small**; write tests alongside code.
- Run `scripts/calibrate_thresholds.py` when threshold tables change and commit updated `thresholds.json`/ROC artifacts.
- Use `experiments/eval/run.py` (or `make rc`) to generate ΔMSE/DM/VaR summaries and refresh memo/gallery outputs.
- Configure detection defaults via `configs/detection.yaml` + `reports/**/calibration/thresholds.json`; override programmatically through `src/fjs/config.get_detection_settings()` when tests require deterministic gating.
- Enable `FJS_DEBUG=1` to stream per-candidate diagnostics (edge margins, admissibility, isolation, angle, a-grid index, accept/reject reason).

### Don’t
- Don’t broaden permissions; don’t fetch external datasets without explicit instruction.
- Don’t replace the full spectrum with de‑aliased values.
- Don’t merge if CI is red or tests/regression artifacts are missing.

## Folder cues
- `src/data/` — loaders & groupers (daily, DoW, Vol‑State)
- `src/fjs/` — MP edge, de‑alias, overlay
- `src/baselines/` — LW, OAS, CC, RIE, EWMA, factors
- `experiments/` — rolling evaluation scripts
- `experiments/eval/run.py` — sprint evaluation harness (ΔMSE, DM, VaR/ES)
- `scripts/calibrate_thresholds.py` — synthetic FPR sweeps (`reports/calibration/thresholds.json`)
- `reports/` — RC artifacts (memo, figures, tables)
- `tools/` — memo & gallery builders

## Permissions (Codex & similar)
- Allowed: read/list files; run `pytest`, `ruff`, `mypy`; run scripts in this repo.
- Ask first: install packages; delete or move files; network access; large downloads.

## Git & commits
- **Conventional Commits** (e.g., `feat(overlay): add eigenvalue substitution`)
- One feature per PR; include tests and docs.
- After each milestone: run tests and commit; include before/after notes in the body.

## Success criteria
- Crisis regime ΔMSE ≥ 3–5% improvement and better VaR/ES; Calm ≈ baseline; synthetic calibration shows FPR ≤ 2% at μ=4–8; `make rc` builds memo + gallery; CI green.
