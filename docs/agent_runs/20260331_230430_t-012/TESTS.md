# Tests

- `bash -lc '. .venv/bin/activate && make test-fast'`
  - status: pass
  - summary: `84 passed, 171 deselected in 22.69s`

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_ff5mom_w126 --exec-mode deterministic'`
  - status: running
  - summary: launched successfully at `2026-03-31 23:06:42 UTC`; early
    monitoring shows the worker remains live and CPU-active; as of
    `2026-03-31 23:40:56 UTC`, `run.json` reports `stage=evaluate` /
    `status=running`, but the exposed file surface is still limited to the
    five init-stage artifacts under
    `reports/rc-t-012/dow-paper-v1_ff5mom_w126/`.

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_noprewhiten.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w126 --exec-mode deterministic'`
  - status: running
  - summary: launched successfully at `2026-03-31 23:06:43 UTC`; early
    monitoring shows the worker remains live and CPU-active; as of
    `2026-03-31 23:40:56 UTC`, `run.json` reports `stage=evaluate` /
    `status=running`, but the exposed file surface is still limited to the
    five init-stage artifacts under
    `reports/rc-t-012/dow-paper-v1_noprewhiten_w126/`.

- `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
  - status: pass
  - summary: validator returned `"valid": true` with no errors or warnings for
    the current in-progress T-012 run log.

- `python - <<'PY' ... interim bundle assertion over reports/_bundles/20260331_234217_T-012_gpt_bundle.zip ... PY`
  - status: fail
  - summary: the first interim T-012 bundle omitted
    `experiments/eval/config.paper_v1_dow_window252.yaml` and
    `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`, which
    exposed a self-description gap in `scripts/make_gpt_bundle.py`.

- `bash -lc '. .venv/bin/activate && pytest -q tests/test_gpt_bundle.py'`
  - status: pass
  - summary: `2 passed in 0.02s` after the minimal bundle-helper patch.

- `python - <<'PY' ... interim bundle assertion over reports/_bundles/20260331_234249_T-012_gpt_bundle.zip ... PY`
  - status: pass
  - summary: verified the rebuilt interim T-012 bundle now contains the daily
    `dow` decision docs, the T-012 ticket, both new `window252` configs, the
    T-012 run log, and the partial `w126` run directories.

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_ff5mom_w126 --exec-mode deterministic'`
  - status: pass on restarted run
  - summary: the first attempt was interrupted externally and preserved under
    `reports/rc-t-012/_preserved_interrupted_20260401_000435/`; the relaunch
    started at `2026-04-01 00:04:58 UTC`, later completed with
    `run.json status=ok`, `stage=complete`, `run.log` ending at
    `2026-04-01T03:03:37.638920+00:00`, and the required `full/*` surfaces plus
    `overlay_toggle.md` present.

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_noprewhiten.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w126 --exec-mode deterministic'`
  - status: pass on restarted run
  - summary: the first attempt was interrupted externally and preserved under
    `reports/rc-t-012/_preserved_interrupted_20260401_000435/`; the relaunch
    started at `2026-04-01 00:04:58 UTC`, later completed with
    `run.json status=ok`, `stage=complete`, `run.log` ending at
    `2026-04-01T03:06:41.892321+00:00`, and the required `full/*` surfaces plus
    `overlay_toggle.md` present.

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_window252.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_ff5mom_w252 --exec-mode deterministic'`
  - status: pass
  - summary: launched successfully at `2026-04-01 03:07:24 UTC`; later
    monitoring confirmed the same long live-but-observability-poor state seen
    in T-008/T-010 while the worker remained CPU-active, and the run then
    completed with `run.json status=ok`, `stage=complete`, `run.log` ending at
    `2026-04-01T05:45:32.547105+00:00`, and the required `full/*` surfaces plus
    `overlay_toggle.md` present.

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w252 --exec-mode deterministic'`
  - status: pass
  - summary: launched successfully at `2026-04-01 03:07:24 UTC`; later
    monitoring confirmed the same long live-but-observability-poor state seen
    in T-008/T-010 while the worker remained CPU-active, and the run then
    completed with `run.json status=ok`, `stage=complete`, `run.log` ending at
    `2026-04-01T05:48:23.079791+00:00`, and the required `full/*` surfaces plus
    `overlay_toggle.md` present.

- `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python tools/make_summary.py --rc-dir ${RC_ROOT}'`
  - status: pass
  - summary: wrote `reports/rc-t-012/summary/summary_detection.csv`,
    `summary_perf.csv`, `summary_skip_stats.csv`, `overlay_forensics.csv`,
    `kill_criteria.json`, `limitations.md`, and `completeness.json`.

- `python - <<'PY' ... T-012 artifact and comparison assertion over reports/rc-t-012 ... PY`
  - status: pass
  - summary: confirmed all four run directories contain the required
    `run.json`, `resolved_config.json`, `full/metrics.csv`,
    `full/diagnostics.csv`, `full/risk.csv`, and `overlay_toggle.md`; the
    shared summary directory contains the required summary surfaces;
    `t012_full_regime_comparison.csv` contains both portfolios for all four
    runs with the mandatory full-regime fields; all four `full/metrics.csv`
    files retain the mandatory baseline rows; and `campaign_decision.md` ends
    with an allowed explicit classification.

- `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
  - status: pass
  - summary: validator returned `"valid": true` with no errors or warnings for
    the completed T-012 run log.

- `python - <<'PY' ... bundle assertion over reports/_bundles/20260401_055651_T-012_gpt_bundle.zip ... PY`
  - status: fail then pass after helper repair
  - summary: the first final bundle build omitted
    `reports/rc-t-010/summary/advisor_decision.md`, which exposed one last
    self-description gap in `scripts/make_gpt_bundle.py`; after the minimal
    helper patch and rebuild, the same assertion passed and confirmed the final
    bundle contains the T-012 ticket, the T-012 run log, the required daily
    `dow` decision docs, the ratified T-010 advisor memo, the T-012 summary
    outputs, and the core outputs from all four completed T-012 run
    directories.

- `bash -lc '. .venv/bin/activate && pytest -q tests/test_gpt_bundle.py'`
  - status: pass
  - summary: `2 passed in 0.04s` after the minimal helper patch that added the
    required T-010 advisor memo to the final T-012 bundle.

- `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
  - status: pass
  - summary: validator returned `"valid": true` again after the final
    bookkeeping patch to `META.json`, `PROGRESS.md`, and the T-012 run-log
    files.

- `python - <<'PY' ... final bundle assertion over reports/_bundles/20260401_055651_T-012_gpt_bundle.zip ... PY`
  - status: pass
  - summary: after rebuilding the same bundle path against the finalized
    run-log state, the zip assertion still passes and confirms the bundle
    remains self-describing as T-012 while containing the required T-010 memo,
    the T-012 run log, the T-012 summary outputs, and the core outputs from
    all four completed T-012 runs.
