# Results
- RC-lite-sanity (EXEC_MODE=deterministic) completed at `reports/rc-20251219-sanity-20251219_050735/`; weekly outputs under `experiments/equity_panel/outputs_rc-lite-20251219_20251219_050735/{dow-weekly,nested}/`.
- `summary_sanity.json` regenerated with completeness metadata; `incomplete_runs` is empty and all sections report `status=complete`. Aggregate detection_rate_mean≈0.0268; accept_share_mean=0.0.
- Daily overlays remain harmful (DoW ΔMSE_EW≈1.24e-10, MV≈4.5e-11; Vol ΔMSE_EW≈3.67e-11). Weekly DoW + nested smoke still 0 detections / accept_share=0.
- `summary/completeness.json` emitted by `tools/make_summary.py`; completeness and cap info now surface in limitations/kill_criteria.
- Working tree dirty before validation run due to ongoing ticket work (branch `ticket-05-rc-sanity-summary-hardening`).
- Note: `docs/DOCS_AND_LOGGING_SYSTEM.md` referenced in prompt is absent in repo; followed existing agent_runs logging format.
- Bundle: `bundles/20251219_044404_ticket-05_rc-sanity-summary-hardening.tar.gz` (manual tarball; `make gpt-bundle` target not present).
