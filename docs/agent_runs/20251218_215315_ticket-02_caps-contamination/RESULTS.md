# Results
Goal: Fix evaluation cap contamination transparency and denominators.
Summary outcome:
- Added run_manifest JSON for daily eval and equity-panel runs (config/dataset hashes, window counts, cap flags, start/end, exec mode) and mirrored manifest into run.json; summary outputs now expose windows_total/evaluated, max_windows, cap_active, and window_coverage.
- make_summary and summarize_rc_sanity now consume manifests, print cap/incomplete reasons, and exclude flagged runs from aggregates (kill-criteria input filtered); detection/percent_changed now reported alongside window coverage.
- Makefile: optional RC_MAX_WINDOWS knob (default off) for rc-lite-sanity daily eval; caps remain off by default for production-ish targets.

Cap/truncation code paths located:
- experiments/eval/run.py: max_windows cap applied in run_evaluation (start_indices slice) and regime sampling via _limit_windows_by_regime (calm_window_sample/crisis_window_top_k).
- experiments/eval/config.py: CLI/config normalisation for max_windows (defaults to None/off).
- tools/make_summary.py & tools/summarize_rc_sanity.py: summary aggregation now reads run manifests, surfaces windows_total/evaluated, cap_active, max_windows, and drops capped/incomplete runs from aggregates.
- Makefile rc-lite-sanity target: optional RC_MAX_WINDOWS flag (default off) passes --max-windows to daily eval.
- experiments/equity_panel/run.py: summary payload + run_manifest now carry rolling_windows_evaluated, gating skips, and window coverage (cap_active always false; incomplete_reason set when skips reduce coverage).

Artifacts:
- Run log: docs/agent_runs/20251218_215315_ticket-02_caps-contamination/
- Cap-off rc-lite-sanity (deterministic): reports/rc-20251218-sanity-20251218_223414 with daily runs {dow-tyler, vol-tyler} and weekly outputs experiments/equity_panel/outputs_rc-lite-20251218_20251218_223414/{dow-weekly,nested}; summary + summary_sanity.json produced.
- Cap-on rc-lite-sanity (max_windows=5): reports/rc-20251218-sanity-20251218_230000 with weekly outputs experiments/equity_panel/outputs_rc-lite-20251218_20251218_230000/{dow-weekly,nested}; summary + summary_sanity.json produced (daily runs flagged cap_active=true).

Key metrics:
- Cap-off summary_sanity (223414): daily_dow detection_rate≈0.055, percent_changed=1.0, window_coverage=1.0; daily_vol detection_rate≈0.052, percent_changed≈0.939, window_coverage=1.0; weekly_dow/nested window_coverage=0.5 (excluded for gating_skip).
- Cap-on summary_sanity (230000): daily runs window_coverage≈0.09 with cap_sources=["max_windows=5"] (excluded); weekly window_coverage=0.5, excluded for gating_skip counts.
- Tests: PATH=.venv/bin:$PATH make test-fast (67 passed, 147 deselected; pytest DeprecationWarning about datetime.utcnow).

Failures / warnings:
- Two initial rc-lite-sanity invocations timed out; ignored partial dirs (reports/rc-20251218-sanity-20251218_222842 and early 223414 run).
- First manual vol rerun for rc-20251218-sanity-20251218_223414 timed out; reran after clean slate.
- Weekly runs in both stamps excluded in summaries due to gating_skip (coverage=0.5).
