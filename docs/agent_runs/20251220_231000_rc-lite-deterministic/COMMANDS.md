RUN_NAME=20251220_231000_rc-lite-deterministic
# Setup
RUN_NAME=$RUN_NAME
mkdir -p docs/agent_runs/$RUN_NAME
echo $RUN_NAME > docs/agent_runs/.current_run_name
cat > docs/agent_runs/$RUN_NAME/PROMPT.md <<'PROMPT'
Task: Deterministic RC-lite run with no opaque diagnostics, uncapped or clearly labeled high coverage, comparison_valid==true for key metrics with nontrivial n_effective. Do it properly and the best way possible.
PROMPT
touch docs/agent_runs/$RUN_NAME/{RESULTS.md,TESTS.md,META.md}
EXEC_MODE=deterministic RC_WORKERS=4 make rc-lite
RC_VERIFY_DATASET='python3 tools/verify_dataset.py data/returns_daily.csv --registry data/registry.json' RC_VERIFY_FACTORS='python3 tools/verify_dataset.py data/factors/ff5mom_daily.csv --registry data/factors/registry.json' EXEC_MODE=deterministic RC_WORKERS=4 make rc-lite
ln -s /usr/bin/python3 /usr/bin/python
EXEC_MODE=deterministic RC_WORKERS=4 make rc-lite
EXEC_MODE=deterministic RC_WORKERS=4 make rc-lite-sanity
PYTHONPATH=src:. OMP_NUM_THREADS=1 EXEC_MODE=deterministic python3 experiments/equity_panel/run.py --config experiments/equity_panel/config.smoke.yaml --design dow --estimator dealias --output-dir experiments/equity_panel/outputs_rc-lite-20251220_20251220_233700/dow-weekly --cache-dir .cache/rc-lite --resume --precompute-panel --gating-mode calibrated --gating-calibration calibration/edge_delta_thresholds.json --edge-mode tyler --prewhiten ff5mom --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv --gating-diagnostics
PYTHONPATH=src:. OMP_NUM_THREADS=1 EXEC_MODE=deterministic python3 experiments/equity_panel/run.py --config experiments/equity_panel/config.nested.smoke.yaml --design nested --estimator dealias --output-dir experiments/equity_panel/outputs_rc-lite-20251220_20251220_233700/nested --cache-dir .cache/rc-lite --resume --precompute-panel --gating-mode calibrated --gating-calibration calibration/edge_delta_thresholds.json --edge-mode tyler --prewhiten ff5mom --use-factor-prewhiten 1 --factor-csv data/factors/ff5mom_daily.csv --gating-diagnostics
