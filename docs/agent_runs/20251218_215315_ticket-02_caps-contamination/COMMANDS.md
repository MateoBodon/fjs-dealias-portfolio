# Commands (chronological)
2025-12-18T21:53:15Z pwd
2025-12-18T21:53:20Z ls
2025-12-18T21:53:25Z git status -sb
2025-12-18T21:53:30Z git checkout -b ticket-02-caps-contamination
2025-12-18T21:53:35Z date +%Y%m%d_%H%M%S
2025-12-18T21:53:40Z mkdir -p docs/agent_runs/20251218_215315_ticket-02_caps-contamination
2025-12-18T21:53:45Z mkdir -p docs/prompts
2025-12-18T21:53:50Z cat > docs/prompts/20251218_215315_ticket-02_caps-contamination.md
2025-12-18T21:53:55Z cp docs/prompts/20251218_215315_ticket-02_caps-contamination.md docs/agent_runs/20251218_215315_ticket-02_caps-contamination/PROMPT.md
2025-12-18T21:54:00Z cat > docs/agent_runs/20251218_215315_ticket-02_caps-contamination/COMMANDS.md
2025-12-18T21:54:05Z cat > docs/agent_runs/20251218_215315_ticket-02_caps-contamination/RESULTS.md
2025-12-18T21:54:10Z cat > docs/agent_runs/20251218_215315_ticket-02_caps-contamination/TESTS.md
2025-12-18T21:54:15Z cat > docs/agent_runs/20251218_215315_ticket-02_caps-contamination/META.json
2025-12-18T21:55:00Z rg "max_windows|cap_first|window_cap|limit_windows|truncate|MAX_WINDOWS" experiments tools Makefile project_state src reports scripts tests
2025-12-18T21:56:00Z rg "max_windows" experiments/equity_panel experiments/eval experiments/daily
2025-12-18T21:57:00Z sed -n '1,240p' experiments/equity_panel/run.py
2025-12-18T21:57:30Z sed -n '1,260p' experiments/eval/config.py
2025-12-18T21:58:00Z sed -n '2320,2465p' experiments/eval/run.py
2025-12-18T21:58:30Z sed -n '2805,3045p' experiments/eval/run.py
2025-12-18T21:59:00Z sed -n '320,520p' tools/make_summary.py
2025-12-18T21:59:30Z sed -n '1,240p' tools/summarize_rc_sanity.py
2025-12-18T22:14:00Z make test-fast
2025-12-18T22:15:00Z pip install -e '.[dev]'
2025-12-18T22:16:00Z python3 -m venv .venv
2025-12-18T22:17:00Z .venv/bin/pip install -e '.[dev]'
2025-12-18T22:20:00Z PATH=.venv/bin:$PATH make test-fast
2025-12-18T22:23:00Z PATH=.venv/bin:$PATH EXEC_MODE=deterministic RC_PY="PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3" make rc-lite-sanity
2025-12-18T22:33:00Z PATH=.venv/bin:$PATH EXEC_MODE=deterministic RC_PY="PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3" make rc-lite-sanity
2025-12-18T22:45:00Z PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3 experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml --design dow --estimator dealias --output-dir experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/dow-weekly --cache-dir .cache/rc-lite --resume --precompute-panel --gating-mode calibrated --gating-calibration calibration/edge_delta_thresholds.json --edge-mode tyler --prewhiten ff5mom --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv
2025-12-18T22:45:20Z PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3 experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --design nested --estimator dealias --output-dir experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/nested --cache-dir .cache/rc-lite --resume --precompute-panel --gating-mode calibrated --gating-calibration calibration/edge_delta_thresholds.json --edge-mode tyler --prewhiten ff5mom --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv
2025-12-18T22:46:00Z PYTHONPATH=src:. .venv/bin/python3 tools/make_summary.py --rc-dir reports/rc-20251218-sanity-20251218_223414
2025-12-18T22:46:40Z PYTHONPATH=src:. .venv/bin/python3 tools/summarize_rc_sanity.py --rc-dir reports/rc-20251218-sanity-20251218_223414 --dow-dir reports/rc-20251218-sanity-20251218_223414/dow-tyler --vol-dir reports/rc-20251218-sanity-20251218_223414/vol-tyler --weekly-dow-dir experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/dow-weekly --nested-dir experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/nested
2025-12-18T22:47:30Z PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3 experiments/eval/run.py --returns-csv data/returns_daily.csv --window 60 --horizon 10 --start 2023-01-01 --end 2023-06-30 --assets-top 50 --group-design vol --group-min-count 3 --group-min-replicates 6 --min-reps-vol 6 --edge-mode tyler --shrinker oas --prewhiten ff5mom --overlay-delta 0.05 --coarse-candidate 1 --gate-mode soft --gate-accept-nonisolated --gate-delta-calibration calibration/edge_delta_thresholds.json --gate-delta-frac-min 0.015 --q-max 2 --mv-gamma 1e-4 --mv-box 0.0,0.1 --mv-turnover-bps 5 --mv-condition-cap 1000000 --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv --out reports/rc-20251218-sanity-20251218_223414/vol-tyler
2025-12-18T22:52:00Z rm -rf reports/rc-20251218-sanity-20251218_223414/vol-tyler
2025-12-18T22:52:30Z PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3 experiments/eval/run.py --returns-csv data/returns_daily.csv --window 60 --horizon 10 --start 2023-01-01 --end 2023-06-30 --assets-top 50 --group-design vol --group-min-count 3 --group-min-replicates 6 --min-reps-vol 6 --edge-mode tyler --shrinker oas --prewhiten ff5mom --overlay-delta 0.05 --coarse-candidate 1 --gate-mode soft --gate-accept-nonisolated --gate-delta-calibration calibration/edge_delta_thresholds.json --gate-delta-frac-min 0.015 --q-max 2 --mv-gamma 1e-4 --mv-box 0.0,0.1 --mv-turnover-bps 5 --mv-condition-cap 1000000 --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv --out reports/rc-20251218-sanity-20251218_223414/vol-tyler
2025-12-18T22:58:00Z PYTHONPATH=src:. .venv/bin/python3 tools/make_summary.py --rc-dir reports/rc-20251218-sanity-20251218_223414
2025-12-18T22:58:20Z PYTHONPATH=src:. .venv/bin/python3 tools/summarize_rc_sanity.py --rc-dir reports/rc-20251218-sanity-20251218_223414 --dow-dir reports/rc-20251218-sanity-20251218_223414/dow-tyler --vol-dir reports/rc-20251218-sanity-20251218_223414/vol-tyler --weekly-dow-dir experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/dow-weekly --nested-dir experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/nested
2025-12-18T22:59:00Z PATH=.venv/bin:$PATH EXEC_MODE=deterministic RC_MAX_WINDOWS=5 RC_PY="PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python3" make rc-lite-sanity
2025-12-18T23:05:00Z git diff > docs/agent_runs/20251218_215315_ticket-02_caps-contamination/DIFF.patch
2025-12-18T23:08:00Z PATH=.venv/bin:$PATH make test-fast
