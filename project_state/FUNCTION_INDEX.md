---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Function & Class Index

Derived from project_state/_generated/symbol_index.json. Paths are repo-relative; line numbers are 1-based.

## ablate.run
Path: `experiments/ablate/run.py`
- Classes:
  - `PanelSpec` (line 58): 
- Functions:
  - `_load_yaml(path)` (line 66): 
  - `_coerce_bool(value)` (line 73): 
  - `_coerce_value(key, value)` (line 87): 
  - `_normalise_defaults(defaults)` (line 99): 
  - `_normalise_combo(combo)` (line 106): 
  - `_combo_identifier(params)` (line 113): 
  - `_is_default_combo(combo, defaults, keys)` (line 129): 
  - `_ensure_dir(path)` (line 150): 
  - `_load_panels(section)` (line 154): 
  - `_extract_perf(perf_df, regime, portfolio)` (line 174): 
  - `_extract_detection(det_df, regime)` (line 183): 
  - `_safe_get(series, key)` (line 191): 
  - `run_ablation(config_path)` (line 200): 
  - `parse_args(argv)` (line 440): 
  - `main(argv)` (line 474): 

## aggregate_runs
Path: `tools/aggregate_runs.py`
- Functions:
  - `_resolve_runs(patterns)` (line 14): 
  - `_load_run_metadata(run_dir)` (line 26): 
  - `aggregate_runs(run_dirs)` (line 48): 
  - `parse_args()` (line 79): 
  - `main()` (line 102): 

## baselines.covariance
Path: `src/baselines/covariance.py`
- Functions:
  - `_symmetrize(matrix)` (line 21): 
  - `sample_covariance(observations)` (line 25): Sample covariance with Bessel correction.
  - `lw_covariance(observations)` (line 38): Ledoit–Wolf shrinkage covariance estimator.
  - `oas_covariance(observations)` (line 44): Oracle Approximating Shrinkage (OAS) estimator.
  - `cc_covariance(observations)` (line 50): Ledoit–Wolf constant-correlation shrinkage estimator.
  - `rie_covariance(sample_covariance)` (line 56): Rotationally-invariant estimator (RIE) shrinkage towards the spectrum mean.
  - `quest_covariance(sample_covariance)` (line 81): QuEST-style spectral clipping based on Marchenko–Pastur support.
  - `ewma_covariance(observations)` (line 112): Exponentially weighted moving-average covariance estimate.

## baselines.factors
Path: `src/baselines/factors.py`
- Classes:
  - `PrewhitenResult` (line 44): 
- Functions:
  - `_normalise_columns(frame)` (line 53): 
  - `_detect_percentage_scale(frame)` (line 61): 
  - `_load_candidate(path)` (line 74): 
  - `load_observed_factors()` (line 105): Load observed factor returns, preferring FF5+MOM datasets when available.
  - `_prepare_design_matrix(index, factors)` (line 158): 
  - `_align_returns_factors(returns, factors)` (line 174): 
  - `prewhiten_returns(returns, factors)` (line 206): Regress asset returns on observed factors and return residual series.

## baselines.test_covariance
Path: `tests/baselines/test_covariance.py`
- Functions:
  - `_make_observations(samples, assets, seed)` (line 17): 
  - `_assert_psd(matrix, atol)` (line 23): 
  - `test_statistical_covariances_are_psd(factory)` (line 30): 
  - `test_ewma_covariance_matches_manual_loop()` (line 38): 
  - `test_rie_covariance_shrinks_towards_mean_spectrum()` (line 52): 
  - `test_quest_covariance_clips_to_mp_support()` (line 66): 

## baselines.test_load_factors
Path: `tests/baselines/test_load_factors.py`
- Functions:
  - `_write_factor_csv(path, scale)` (line 12): 
  - `test_load_observed_factors_prefers_explicit_path(tmp_path)` (line 26): 
  - `test_load_observed_factors_builds_proxy(tmp_path)` (line 40): 
  - `test_load_observed_factors_auto_scales_percentages(tmp_path)` (line 55): 
  - `test_load_observed_factors_errors_without_returns(tmp_path)` (line 63): 

## baselines.test_prewhiten
Path: `tests/baselines/test_prewhiten.py`
- Functions:
  - `_simulated_returns(rng)` (line 13): 
  - `test_prewhiten_reduces_spike_strength()` (line 37): 
  - `test_prewhiten_residuals_preserve_null_fpr()` (line 64): 
  - `test_prewhiten_result_exposes_betas_and_intercepts()` (line 99): 

## bench_linalg
Path: `scripts/bench_linalg.py`
- Functions:
  - `parse_args(argv)` (line 25): 
  - `_configure_threads(threads)` (line 50): 
  - `_detect_instance_type()` (line 68): 
  - `_blas_info()` (line 80): 
  - `_bench_dimension(dim, repeats, rng)` (line 93): 
  - `main(argv)` (line 112): 
  - `_git_sha()` (line 157): 

## build_brief
Path: `tools/build_brief.py`
- Functions:
  - `_format_percent(value)` (line 31): 
  - `_safe_float(value)` (line 37): 
  - `_aggregate_reason_table(reason_df)` (line 44): 
  - `build_brief(config_path)` (line 69): 
  - `parse_args()` (line 232): 
  - `main()` (line 243): 

## build_gallery
Path: `tools/build_gallery.py`
- Functions:
  - `_load_config(path)` (line 31): 
  - `_discover_run_paths(entries)` (line 37): 
  - `_gather_rejections(summary_df, run_tag)` (line 66): 
  - `_load_ablation(run_path)` (line 80): 
  - `_edge_dataframe(rolling_df, run_tag)` (line 91): 
  - `build_gallery(config_path)` (line 106): 
  - `parse_args()` (line 177): 
  - `main()` (line 188): 

## build_memo
Path: `tools/build_memo.py`
- Functions:
  - `_load_config(path)` (line 30): 
  - `_discover_run_paths(entries)` (line 35): 
  - `_latest_summary_dir(root)` (line 64): 
  - `_load_summary_artifacts(config)` (line 74): 
  - `_format_kill_criteria_payload(kill_data)` (line 143): 
  - `_markdown_table(df)` (line 181): 
  - `_format_percent(value)` (line 200): 
  - `_pick_strategy_row(group, strategies)` (line 206): 
  - `_format_delta(value)` (line 217): 
  - `_format_ci(lo, hi)` (line 228): 
  - `_format_edge_metric(value)` (line 234): 
  - `_format_pvalue(value)` (line 242): 
  - `_format_detection(value)` (line 250): 
  - `_format_windows(value)` (line 256): 
  - `_numeric(row, key)` (line 262): 
  - `_row_for(frame, regime, portfolio)` (line 271): 
  - `_prettify_reason(reason)` (line 286): 
  - `_collect_rejection_records(summary_df, run_tag)` (line 290): 
  - `_build_key_tables(panel_df)` (line 301): 
  - `_build_rejection_tables(rejection_records)` (line 528): 
  - `build_memo(config_path)` (line 556): 
  - `parse_args()` (line 1297): 
  - `main()` (line 1308): 

## clean_outputs
Path: `tools/clean_outputs.py`
- Functions:
  - `is_tagged_directory(path)` (line 12): 
  - `unique_destination(base)` (line 16): 
  - `collect_legacy(root)` (line 29): 
  - `clean_outputs(root)` (line 40): 
  - `parse_args()` (line 78): 
  - `main()` (line 101): 

## daily.config
Path: `experiments/daily/config.py`
- Classes:
  - `DailyDesign` (line 9): Configuration defaults for a replicated daily experiment.

## daily.grouping
Path: `experiments/daily/grouping.py`
- Classes:
  - `GroupingError` (line 16): Raised when a sliding window cannot be balanced for replicates.
- Functions:
  - `_ensure_datetime_index(frame)` (line 20): 
  - `group_by_week(frame)` (line 25): Balance a window by complete weeks (default: Monday-aligned business weeks).
  - `group_by_day_of_week(frame)` (line 47): Balance a window by Day-of-Week replicates across complete weeks.
  - `_vol_state_codes(frame, vol_proxy)` (line 95): 
  - `group_by_vol_state(frame)` (line 122): Balance a window by volatility-state buckets (calm/mid/crisis).
  - `group_by_dow_vol(frame)` (line 175): Balance windows by crossing day-of-week with volatility states.
  - `group_by_dow_month(frame)` (line 225): Balance windows by crossing day-of-week with calendar month.

## daily.run
Path: `experiments/daily/run.py`
- Functions:
  - `_detect_forward_override(argv, flag)` (line 14): 
  - `parse_args(argv)` (line 28): 
  - `_default_out(design, rc_date)` (line 78): 
  - `main(argv)` (line 84): 

## equity_panel.run
Path: `experiments/equity_panel/run.py`
- Classes:
  - `PreparedWindowStats` (line 152): 
- Functions:
  - `_parse_box_bounds(bounds)` (line 129): Normalise min-variance box bounds into a (lo, hi) tuple.
  - `_load_prepared_from_cache(cached_stats, design_mode, y_fit_raw, code_signature_hash, expected_nested_replicates)` (line 167): 
  - `_compute_oneway_prepared(fit_blocks, y_fit_raw, replicates, code_signature_hash)` (line 270): 
  - `_compute_nested_prepared(fit_blocks, y_fit_raw, expected_reps, code_signature_hash)` (line 339): 
  - `_compute_grouped_design_prepared(y_fit_raw, group_labels)` (line 635): Compute MANOVA statistics for alternate balanced daily groupings.
  - `_prepare_window_stats(design_mode, fit_blocks, replicates)` (line 709): 
  - `_infer_skip_reason(diag_local)` (line 739): Best-effort categorisation for windows with no accepted detections.
  - `load_config(path)` (line 783): Load experiment configuration, falling back to defaults.
  - `_generate_synthetic_prices(path)` (line 806): Create a synthetic price panel for quick smoke testing.
  - `_mp_edges(noise_variance, n_assets, n_samples)` (line 831): Return approximate Marčenko–Pastur bulk edges.
  - `_prepare_data(config)` (line 843): Load daily returns from returns CSV or derive from prices CSV.
  - `_apply_preprocessing(daily_returns)` (line 868): Apply optional robustness preprocessing to daily returns.
  - `_run_param_ablation(daily_returns, output_dir)` (line 887): Grid sweep over detection parameters; emit CSV and heatmaps (E5).
  - `_load_or_build_balanced_panel(daily_returns)` (line 1127): Load a cached balanced panel or build a fresh one from daily returns.
  - `_run_single_period(daily_returns)` (line 1193): Execute the rolling evaluation for a single date range.
  - `_run_sigma_ablation(daily_returns, output_dir, cs_drop_top_frac, delta, delta_frac, eps, stability_eta, signed_a, target_component)` (line 3225): Evaluate Cs perturbations and persist sensitivity diagnostics.
  - `run_experiment(config_path)` (line 3311): Execute the rolling equity forecasting experiment.
  - `main()` (line 3756): Entry point for CLI execution.

## equity_panel.sweep_acceptance
Path: `experiments/equity_panel/sweep_acceptance.py`
- Classes:
  - `SweepParams` (line 42): 
- Functions:
  - `_load_grid(arg)` (line 58): 
  - `_build_parameter_grid(grid_cfg)` (line 80): 
  - `_load_factor_returns(config)` (line 99): 
  - `_extract_metrics(run_dir, estimators)` (line 124): 
  - `run_sweep(args)` (line 197): 
  - `parse_args(argv)` (line 359): 
  - `main(argv)` (line 392): 

## etf_panel.run
Path: `experiments/etf_panel/run.py`
- Classes:
  - `ETFConfig` (line 20): 
- Functions:
  - `parse_args(argv)` (line 30): 
  - `main(argv)` (line 82): 

## eval.balance
Path: `src/eval/balance.py`
- Classes:
  - `BalanceTelemetry` (line 13): 
  - `BalanceResult` (line 25): 
- Functions:
  - `build_balanced_window(frame, group_labels)` (line 32): 

## eval.clean
Path: `src/eval/clean.py`
- Classes:
  - `NaNPolicyTelemetry` (line 13): 
  - `NaNPolicyResult` (line 23): 
- Functions:
  - `apply_nan_policy(frame, group_labels)` (line 29): 

## eval.config
Path: `experiments/eval/config.py`
- Classes:
  - `ResolveResult` (line 110): 
- Functions:
  - `_deep_merge(base, override)` (line 15): 
  - `_load_yaml(path)` (line 25): 
  - `_load_json(path)` (line 35): 
  - `_normalise_layer(payload)` (line 45): 
  - `resolve_eval_config(args)` (line 115): 

## eval.diagnostics
Path: `experiments/eval/diagnostics.py`
- Classes:
  - `DiagnosticReason` (line 6): 

## eval.inject_spike
Path: `experiments/eval/inject_spike.py`
- Classes:
  - `WindowSample` (line 28): 
- Functions:
  - `_parse_float_list(raw, name)` (line 34): 
  - `_make_overlay_config(config)` (line 49): 
  - `_collect_windows(config, raw_returns, residuals, vol_proxy_full)` (line 75): 
  - `_baseline_fp(samples, overlay_cfg)` (line 160): 
  - `_inject_spike(matrix, rng, mu)` (line 169): 
  - `parse_args(argv)` (line 182): 
  - `main(argv)` (line 210): 

## eval.run
Path: `experiments/eval/run.py`
- Classes:
  - `EvalOutputs` (line 85): 
  - `EvalConfig` (line 224): 
- Functions:
  - `_plot_regime_histograms(diagnostics_df, column)` (line 97): 
  - `_format_group_label_counts(labels, design)` (line 288): 
  - `_vol_state_label(value, calm_cut, crisis_cut)` (line 329): 
  - `_serialise_config(config)` (line 339): 
  - `_current_git_sha()` (line 406): 
  - `_write_run_metadata(path, payload)` (line 418): 
  - `_paths_to_strings(path_map)` (line 423): 
  - `_mode_string(values)` (line 427): 
  - `_aligned_error_table(metrics, regime, portfolio)` (line 440): 
  - `_aligned_dm_stat(metrics, regime, portfolio)` (line 468): 
  - `_apply_multi_alignment_guard(detections)` (line 497): Drop detections beyond the first if alignment cosine falls below threshold.
  - `_sign_test_stat(aligned, comparator)` (line 525): 
  - `_bootstrap_delta_mse(diffs, resamples, rng, block_size)` (line 549): 
  - `_vol_thresholds(vol_proxy, train_end, config)` (line 583): 
  - `parse_args(argv)` (line 598): 
  - `_compute_vol_proxy(returns, span)` (line 1041): 
  - `_write_overlay_toggle(path, summary)` (line 1049): 
  - `_plot_histogram(series, path)` (line 1081): 
  - `_plot_acceptance_edge_histograms(diagnostics_df, design, out_dir)` (line 1108): 
  - `_detail_defaults()` (line 1138): 
  - `_safe_nanmean(values)` (line 1157): 
  - `_safe_nanmedian(values)` (line 1167): 
  - `_top_mean(values, count)` (line 1177): 
  - `_safe_share(successes, total)` (line 1188): 
  - `_required_replicates(design, config)` (line 1194): 
  - `_build_grouped_window(frame)` (line 1205): 
  - `_min_variance_weights(covariance)` (line 1240): 
  - `_expected_shortfall(sigma, alpha)` (line 1314): 
  - `_realised_tail_mean(returns, var_threshold)` (line 1319): 
  - `_safe_condition_number(matrix)` (line 1326): 
  - `_qlike_loss(forecast_var, realised_var)` (line 1333): 
  - `_limit_windows_by_regime(metrics_df, diagnostics_df)` (line 1345): 
  - `_window_regime(vol_proxy, date, calm_cut, crisis_cut)` (line 1405): 
  - `_prepare_returns(config)` (line 1433): 
  - `run_evaluation(config)` (line 1538): 
  - `main(argv)` (line 3562): 

## eval.sensitivity
Path: `experiments/eval/sensitivity.py`
- Classes:
  - `Combo` (line 32): 
- Functions:
  - `_parse_bool_grid(raw)` (line 51): 
  - `_parse_alignment_grid(raw)` (line 66): 
  - `_parse_float_grid(raw, name)` (line 84): 
  - `_ensure_matplotlib()` (line 99): 
  - `_thread_env()` (line 106): 
  - `_run_evaluation(command, env)` (line 128): 
  - `_build_command(base_args, combo, run_dir)` (line 132): 
  - `_load_first_row(path)` (line 186): 
  - `_changed_window_ids(detail_source)` (line 193): 
  - `_mean_delta_sq_error(metrics_path, changed_ids, portfolio)` (line 211): 
  - `_dm_stats_from_metrics(metrics_path, changed_ids, portfolio)` (line 233): 
  - `_plot_heatmap(subset, delta_values, stability_values, metric, title, xlabel, ylabel, path)` (line 265): 
  - `_full_regime(detail_df)` (line 305): 
  - `_metric_series(detail_df, column, changed_ids)` (line 315): 
  - `_median_metric(detail_df, column, changed_ids)` (line 334): 
  - `_plot_metric_histograms(metric_map)` (line 341): 
  - `parse_args(argv)` (line 379): 
  - `main(argv)` (line 413): 

## eval.test_balance
Path: `tests/eval/test_balance.py`
- Functions:
  - `test_build_balanced_window_trims_to_min_replicates()` (line 9): 
  - `test_build_balanced_window_intersects_assets_across_groups()` (line 30): 
  - `test_build_balanced_window_flags_insufficient_replicates()` (line 49): 

## eval.test_clean
Path: `tests/eval/test_clean.py`
- Functions:
  - `test_apply_nan_policy_drops_assets_above_threshold()` (line 9): 
  - `test_apply_nan_policy_drops_rows_with_zero_tolerance()` (line 34): 

## evaluation
Path: `src/evaluation/__init__.py`
- Functions:
  - `check_dealiased_applied(estimates)` (line 21): Assert de-aliased forecasts differ from aliased when detections exist.

## evaluation.dm
Path: `src/evaluation/dm.py`
- Functions:
  - `_newey_west_long_run_variance(diffs, lags)` (line 10): 
  - `dm_test(err1, err2)` (line 27): Diebold–Mariano test for equal predictive accuracy.

## evaluation.evaluate
Path: `src/evaluation/evaluate.py`
- Classes:
  - `DeltaSummary` (line 367): 
- Functions:
  - `iqr(values)` (line 32): Interquartile range (75th - 25th percentile).
  - `sign_test_pvalue(differences)` (line 53): Two-sided sign test p-value for paired differences.
  - `qlike(forecasts, realised)` (line 86): Quasi-likelihood (QLIKE) loss for variance forecasts.
  - `block_bootstrap_ci_median(series)` (line 102): Moving block bootstrap CI for the median of a time series.
  - `_clip_prob(value, eps)` (line 163): 
  - `kupiec_pof_test(violations, alpha)` (line 167): Return the Kupiec proportion-of-failures p-value.
  - `christoffersen_independence_test(violations)` (line 184): Return the Christoffersen independence test p-value.
  - `expected_shortfall_test(losses, es_forecasts, violations)` (line 214): Approximate two-sided t-test comparing realised losses and ES forecasts.
  - `alignment_diagnostics(covariance, direction)` (line 239): Return (angle_deg, energy_mu) between detection direction and PCA subspace.
  - `plot_variance_error_panel(errors, base_path)` (line 274): Plot E3: variance MSE mean and distribution by method.
  - `plot_coverage_error(coverage_errors, base_path)` (line 330): Plot E4: VaR(95%) coverage errors by method.
  - `summarize_deltas(deltas)` (line 375): Return robust summary statistics and CI for paired deltas.
  - `build_metrics_summary()` (line 402): Aggregate window-level errors into a metrics summary DataFrame.

## evaluation.factor
Path: `src/evaluation/factor.py`
- Classes:
  - `POETResult` (line 42): 
- Functions:
  - `observed_factor_covariance(returns, factors)` (line 13): Estimate Σ = B Σ_f Bᵀ + Σ_ε from observed factor returns via cross-sectional OLS.
  - `_poet_ic(residual_var, k, p, n)` (line 47): 
  - `poet_lite_covariance(returns)` (line 53): Estimate a POET-lite covariance using PCA loadings with simple residual shrinkage.

## experiments.test_ablate_run
Path: `tests/experiments/test_ablate_run.py`
- Functions:
  - `_write_returns(path, rows, cols)` (line 13): 
  - `_write_config(path, returns_path)` (line 22): 
  - `test_run_ablation_small_grid(tmp_path)` (line 72): 

## experiments.test_daily_grouping
Path: `tests/experiments/test_daily_grouping.py`
- Functions:
  - `_make_returns_frame(start, periods)` (line 17): 
  - `test_group_by_day_of_week_balances()` (line 24): 
  - `test_group_by_day_of_week_requires_full_weeks()` (line 35): 
  - `test_group_by_vol_state_balances()` (line 41): 
  - `test_group_by_vol_state_enforces_min_replicates()` (line 63): 
  - `test_daily_cli_forwards_defaults(tmp_path, monkeypatch)` (line 81): 
  - `test_daily_cli_group_alias(tmp_path, monkeypatch)` (line 112): 
  - `test_daily_cli_group_conflict(tmp_path)` (line 138): 
  - `test_daily_cli_forwards_prewhiten(tmp_path, monkeypatch)` (line 154): 
  - `test_daily_cli_forwards_use_factor_prewhiten(tmp_path, monkeypatch)` (line 182): 
  - `test_daily_cli_forwards_assets_top(tmp_path, monkeypatch)` (line 208): 
  - `test_group_by_day_of_week_three_year_slice()` (line 234): 

## experiments.test_eval_run
Path: `tests/experiments/test_eval_run.py`
- Functions:
  - `_make_returns_csv(tmp_path)` (line 27): 
  - `_make_factors_csv(tmp_path)` (line 37): 
  - `test_run_evaluation_emits_artifacts(tmp_path_factory)` (line 50): 
  - `test_run_evaluation_prewhiten_off(tmp_path_factory)` (line 196): 
  - `test_apply_multi_alignment_guard_respects_threshold()` (line 222): 
  - `test_sign_test_stat_filters_ties()` (line 236): 
  - `test_run_evaluation_respects_assets_top(tmp_path_factory)` (line 247): 
  - `test_run_evaluation_vol_design_logs_state(tmp_path_factory)` (line 274): 
  - `test_vol_run_outputs_flip_and_prewhiten_effect(tmp_path_factory)` (line 322): 
  - `test_resolve_eval_config_precedence(tmp_path_factory)` (line 406): 
  - `test_fpr_guard_for_calibrated_thresholds(tmp_path_factory)` (line 485): 
  - `test_resolve_eval_config_prewhiten_cli(tmp_path_factory)` (line 503): 
  - `test_resolve_eval_config_shrinkers(tmp_path_factory, shrinker)` (line 520): 
  - `test_load_daily_panel_from_parquet(tmp_path_factory)` (line 533): 
  - `test_min_variance_weights_turnover_penalty()` (line 554): 
  - `test_dm_alignment_uses_common_windows()` (line 573): 
  - `test_vol_thresholds_use_training_data(tmp_path_factory)` (line 620): 
  - `test_run_evaluation_is_reproducible(tmp_path_factory)` (line 645): 
  - `test_bootstrap_bands_populate_for_overlay(tmp_path_factory)` (line 707): 

## experiments.test_gating_diagnostics
Path: `tests/experiments/test_gating_diagnostics.py`
- Functions:
  - `test_gating_diagnostics_artifact(tmp_path)` (line 10): 

## experiments.test_prewhiten_utils
Path: `tests/experiments/test_prewhiten_utils.py`
- Functions:
  - `_mock_returns(rows, cols)` (line 9): 
  - `_mock_factors(rows)` (line 16): 
  - `test_apply_prewhitening_off_mode_returns_identity()` (line 24): 
  - `test_apply_prewhitening_with_factors_uses_requested_mode()` (line 33): 
  - `test_apply_prewhitening_custom_mode_preserves_columns()` (line 44): 

## experiments.test_skip_reasons
Path: `tests/experiments/test_skip_reasons.py`
- Functions:
  - `test_infer_skip_reason_calibration_missing()` (line 4): 
  - `test_infer_skip_reason_prefers_stability()` (line 13): 

## finance.design
Path: `src/finance/design.py`
- Functions:
  - `build_design_matrix(returns, factors)` (line 7): Join returns with factor realisations on their common timeline.
  - `groups_from_weeks(index)` (line 34): Assign an integer group id to each timestamp based on its week.

## finance.eval
Path: `src/finance/eval.py`
- Functions:
  - `rolling_windows(panel, window_weeks, horizon_weeks)` (line 21): Yield expanding fit/hold windows over the weekly panel.
  - `risk_metrics(forecasts, realised)` (line 52): Compute mean squared error and 95% VaR coverage error.
  - `oos_variance_forecast(y_fit, y_hold, w, estimator)` (line 84): Compute out-of-sample variance forecasts and realised variance.
  - `weekly_cov_from_components(ms1, ms2, replicates, mu_hats, vecs, clip_top)` (line 223): Construct the weekly covariance of summed daily returns from MANOVA components.
  - `variance_forecast_from_components(y_fit, y_hold, replicates, w, detections)` (line 298): Forecast portfolio variance from balanced MANOVA components and compare to realised.
  - `evaluate_portfolio(returns, weights)` (line 400): Compute realised return and volatility for the supplied weights.

## finance.factors
Path: `src/finance/factors.py`
- Functions:
  - `_align_frames(returns, factors, industry)` (line 10): Align inputs on their shared date index and drop factor-side NaNs.
  - `_prepare_design(factors, industry)` (line 43): Combine factor and industry data into a single numeric design matrix.
  - `factor_covariance(R_df, F_df)` (line 64): Estimate an observed-factor covariance matrix via cross-sectional OLS.

## finance.io
Path: `src/finance/io.py`
- Functions:
  - `load_prices_csv(path)` (line 15): Load a tidy price history CSV with canonical dtypes.
  - `to_daily_returns(price_frame)` (line 52): Convert tidy prices to a wide matrix of daily log returns.
  - `load_market_data(path)` (line 79): Backward-compatible alias for :func:`load_prices_csv`.
  - `load_returns_csv(path)` (line 99): Load a tidy daily returns CSV into a wide date-indexed matrix.

## finance.ledoit
Path: `src/finance/ledoit.py`
- Functions:
  - `lw_cov(x)` (line 10): Compute the Ledoit–Wolf covariance estimate.
  - `ledoit_wolf_shrinkage(x)` (line 35): Backward-compatible alias for :func:`lw_cov`.

## finance.loader
Path: `src/finance/loader.py`
- Classes:
  - `WeeklyLoadResult` (line 14): 
- Functions:
  - `_balanced_weekly_from_daily(daily_returns, replicates)` (line 20): Internal helper: balanced weekly panel with a fixed universe.
  - `load_weekly_from_daily_csv(path)` (line 77): Load daily prices CSV, build balanced weekly panel, and print counters.
  - `rolling_windows_fixed_universe(weekly)` (line 108): Yield (fit, hold) windows with per-window fixed-universe enforcement.

## finance.portfolio
Path: `src/finance/portfolio.py`
- Classes:
  - `MinVarMemo` (line 58): Cache penalised covariance factorizations per window.
- Functions:
  - `_symmetrize(matrix)` (line 11): 
  - `_project_box_sum(v, lo, hi, target)` (line 15): 
  - `minvar_ridge_box(Sigma)` (line 88): Projected-gradient minimum-variance solver with ridge and box bounds.
  - `turnover(w_prev, w_new)` (line 164): Compute one-way turnover between consecutive portfolios.
  - `apply_turnover_cost(var_series, w_series, bps)` (line 174): Apply turnover costs (in basis points) to a variance or PnL series.

## finance.portfolios
Path: `src/finance/portfolios.py`
- Classes:
  - `MissingSolverError` (line 10): Raised when a required optimisation solver dependency is unavailable.
  - `OptimizationResult` (line 31): Result container for portfolio optimisation routines.
- Functions:
  - `_get_cvxpy()` (line 14): 
  - `equal_weight(p)` (line 43): Return the equal-weight vector for ``p`` assets.
  - `_solve_min_variance_cvxpy(covariance)` (line 62): Solve the minimum-variance problem using cvxpy.
  - `minimum_variance(covariance)` (line 130): Solve the minimum-variance problem using cvxpy (if available).
  - `min_variance_box(covariance, lb, ub)` (line 148): Solve the minimum-variance problem with box constraints.
  - `optimize_portfolio(covariance, target_return)` (line 169): Return the minimum-variance portfolio; fail loud if solver is missing by default.

## finance.returns
Path: `src/finance/returns.py`
- Functions:
  - `compute_log_returns(prices)` (line 11): Compute log returns for a wide price DataFrame.
  - `weekly_panel(daily_returns, start, end)` (line 35): Aggregate daily log returns into weekly (Monday-start) log returns.
  - `balance_weeks(panel)` (line 80): Create a balanced week/day design from daily returns.

## finance.robust
Path: `src/finance/robust.py`
- Functions:
  - `winsorize(returns_df, q)` (line 7): Clip each column of ``returns_df`` to its [q, 1-q] empirical quantiles.
  - `huberize(returns_df, c)` (line 20): Apply column-wise Huber clipping using median and MAD scale.
  - `tyler_shrink_covariance(observations)` (line 41): Return a Tyler M-estimator with ridge regularisation for positive definiteness.

## finance.shrinkage
Path: `src/finance/shrinkage.py`
- Functions:
  - `_validate_input(R)` (line 13): 
  - `_sample_covariance(X)` (line 22): 
  - `_symmetrize(matrix)` (line 28): 
  - `_warn_and_fill_nonfinite(name, data)` (line 32): 
  - `_assert_psd_and_symmetric(name, matrix)` (line 50): 
  - `oas_covariance(R)` (line 58): Oracle Approximating Shrinkage covariance targeting the identity matrix.
  - `cc_covariance(R)` (line 70): Ledoit–Wolf constant-correlation shrinkage covariance estimator.

## fjs
Path: `src/fjs/__init__.py`
- Functions:
  - `_missing_matplotlib()` (line 12): 

## fjs.balanced
Path: `src/fjs/balanced.py`
- Classes:
  - `BalancedConfig` (line 11): Configuration for the balanced risk contribution solver.
- Functions:
  - `_validate_balanced_inputs(y, groups)` (line 30): Return validated observations and grouping assignments for a balanced design.
  - `_compute_group_means(observations, inverse, counts)` (line 56): Accumulate per-group means using the grouping inverse index.
  - `group_means(y, groups)` (line 69): Compute per-group and overall means for a balanced one-way MANOVA design.
  - `mean_squares(y, groups)` (line 95): Estimate balanced one-way MANOVA mean squares and covariance components.
  - `compute_balanced_weights(returns, config)` (line 152): Compute portfolio weights that balance the contribution of estimated MANOVA spikes.

## fjs.balanced_nested
Path: `src/fjs/balanced_nested.py`
- Classes:
  - `NestedDesignMetadata` (line 11): Balanced nested Year⊃Week design metadata.
- Functions:
  - `_validate_labels(labels, name, expected_length)` (line 24): 
  - `mean_squares_nested(y, year_labels, week_of_year_labels, replicates)` (line 37): Compute balanced nested Year⊃Week MANOVA mean squares.

## fjs.dealias
Path: `src/fjs/dealias.py`
- Classes:
  - `DesignParams` (line 16): 
  - `Detection` (line 24): 
  - `DealiasingResult` (line 61): Container for the results of spectral de-aliasing.
- Functions:
  - `_compute_admissible_root(lam_val, a_vec, C_for_mp, d_vec, n_total, cs_vec)` (line 69): 
  - `_orthonormal_tangent_basis(a_vec)` (line 91): Return an orthonormal basis for the tangent space at ``a_vec`` on the sphere.
  - `_rotate_on_sphere(base, tangent, angle)` (line 119): 
  - `_generate_unit_vectors(component_count, a_grid)` (line 131): 
  - `_normalise_angle(theta)` (line 182): 
  - `_angle_key(theta)` (line 190): 
  - `_sigma_of_a_from_MS(a, MS_list)` (line 195): Return Σ̂(a)=∑_s a_s MS_s (balanced design).
  - `dealias_covariance(covariance, spectrum)` (line 216): Remove aliasing artefacts from a sample covariance matrix.
  - `_validate_inputs(y, groups)` (line 310): 
  - `_default_design(stats)` (line 324): 
  - `_merge_detections(detections, eps_factor)` (line 343): 
  - `dealias_search(y, groups, target_r)` (line 394): Perform Algorithm 1 de-aliasing search for one-way balanced designs.

## fjs.gating
Path: `src/fjs/gating.py`
- Functions:
  - `_as_float(value)` (line 15): Best-effort conversion to float with NaN fallback.
  - `_score_detection(det)` (line 31): Return score tuple (primary score, edge margin, lambda) for ordering.
  - `count_isolated_outliers(eigs, edge, stability)` (line 51): Count isolated spikes relative to the MP edge and stability.
  - `select_top_k(detections, k)` (line 109): Select the top-k detections ranked by score = energy * stability.
  - `_load_delta_thresholds(path_str)` (line 136): Load the calibrated delta thresholds JSON with basic validation.
  - `lookup_calibrated_delta(edge_mode, p, t)` (line 149): Return the calibrated delta_frac for the given (edge_mode, p, t) combo.

## fjs.mp
Path: `src/fjs/mp.py`
- Classes:
  - `MarchenkoPasturModel` (line 94): Summary statistics for a Marchenko–Pastur limiting law.
- Functions:
  - `configure_mp_cache(directory)` (line 27): Configure the on-disk MP edge cache directory at runtime.
  - `clear_mp_cache()` (line 40): Clear the in-memory MP edge cache.
  - `_cache_get(key)` (line 46): 
  - `_cache_set(key, value)` (line 67): 
  - `_hash_arrays()` (line 81): 
  - `_prepare_inputs(a, C, d, N)` (line 112): 
  - `_prepare_cs(Cs, template)` (line 133): 
  - `estimate_Cs_from_MS(MS_list, d_list, c_list, drop_top)` (line 147): Estimate trace-based noise plug-ins C_s from the supplied mean squares.
  - `_k_values(a, C, d, N)` (line 211): 
  - `z_of_m(m, a, C, d, N, Cs)` (line 220): Evaluate the closed-form Marčenko–Pastur z(m) transform.
  - `z0(m, a, C, d, N, Cs)` (line 254): Balanced one-way z0(m) in closed form.
  - `_dz_dm(m, k_vals, numerators)` (line 277): 
  - `_d2z_dm2(m, k_vals, numerators)` (line 290): 
  - `z0_prime(m, a, C, d, N, Cs)` (line 303): Closed-form first derivative z0'(m) for balanced one-way design.
  - `z0_double_prime(m, a, C, d, N, Cs)` (line 325): Closed-form second derivative z0''(m) for balanced one-way design.
  - `_logspace_grid()` (line 346): 
  - `_augment_with_singularities(grid, k_vals)` (line 351): 
  - `_newton_refine(x0, f, fp)` (line 367): One-step Newton refinement with simple safeguards.
  - `_stationary_points(k_vals, numerators)` (line 398): Locate stationary points of z(m) by bracketing zeros of z'(m).
  - `_bisect(func, left, right)` (line 428): 
  - `_crosses_pole(m1, m2, k_vals)` (line 479): Return True if the interval [m1, m2] crosses a pole 1 + k m = 0.
  - `_root_brackets(func, points, k_vals)` (line 491): 
  - `_brackets_sign_change(func, points, k_vals)` (line 525): Find sign-change brackets while guarding against poles.
  - `mp_edge(a, C, d, N, Cs)` (line 558): Locate the upper bulk edge of the Marčenko--Pastur distribution.
  - `_mp_edge_impl(a_arr, c_arr, d_arr, n_float, cs_arr)` (line 579): 
  - `m_edge(a, C, d, N, Cs)` (line 617): Return m_plus where z'(m_plus)=0 and z''(m_plus)<0 (upper edge).
  - `admissible_m_from_lambda(lam, a, C, d, N, Cs)` (line 644): Recover the admissible real root of z(m) = λ with positive slope.
  - `_admissible_m_from_lambda_impl(lam_val, a_arr, c_arr, d_arr, n_float, cs_arr)` (line 669): 
  - `_normalise_order(order, n_strata)` (line 735): 
  - `t_vec(lam, a, C, d, N, c, order, Cs)` (line 760): Evaluate the t-vector associated with λ using the admissible root m(λ).
  - `marchenko_pastur_edges(model)` (line 800): Compute the theoretical support edges for a Marchenko–Pastur distribution.
  - `marchenko_pastur_pdf(model, grid)` (line 817): Evaluate the Marchenko–Pastur density over a grid.
  - `scale_Cs(Cs, alpha)` (line 839): Return a scaled copy of the Cs plug-ins by factor ``alpha``.

## fjs.overlay
Path: `src/fjs/overlay.py`
- Classes:
  - `OverlayConfig` (line 24): 
- Functions:
  - `_bracket_status_label(detections)` (line 52): 
  - `_summarise_pre_gate(detections, cfg)` (line 73): 
  - `_coarse_candidates(observations, cfg)` (line 108): 
  - `_baseline_covariance(sample_covariance)` (line 193): 
  - `_resolve_delta_frac(cfg, observations, groups)` (line 236): 
  - `_gate_detections(detections, cfg, soft_cap, delta_frac_used)` (line 256): 
  - `detect_spikes(observations, groups)` (line 324): 
  - `apply_overlay(sample_covariance, detections)` (line 400): 

## fjs.robust
Path: `src/fjs/robust.py`
- Functions:
  - `_ensure_2d(x)` (line 20): 
  - `_symmetrize(matrix)` (line 28): 
  - `_initial_scatter(x)` (line 32): 
  - `tyler_scatter(observations)` (line 47): Return the Tyler fixed-point scatter estimate with optional ridge regularisation.
  - `huber_scatter(observations, c)` (line 109): Compute a Huber-type reweighted scatter estimator.
  - `edge_from_scatter(scatter, n_features, n_samples)` (line 178): Estimate the upper Marčenko–Pastur edge from a scatter matrix.

## fjs.spectra
Path: `src/fjs/spectra.py`
- Functions:
  - `topk_eigh(matrix, k)` (line 16): Return the largest ``k`` eigenpairs of a symmetric matrix.
  - `project_alignment(vector, subspace)` (line 48): Compute the projection norm of ``vector`` onto the span of ``subspace``.
  - `_ensure_path(path)` (line 80): 
  - `plot_spectrum_with_edges(eigenvalues, edges, out_path)` (line 86): Plot an empirical spectrum together with optional reference edge lines.
  - `plot_spike_timeseries(time_index, aliased_series, dealiased_series, out_path)` (line 161): Plot aliased and de-aliased spike estimates against a time index.
  - `estimate_spectrum(eigenvalues)` (line 219): Return a sorted copy of ``eigenvalues`` (placeholder estimator).

## fjs.test_overlay
Path: `tests/fjs/test_overlay.py`
- Functions:
  - `_make_detection(mu, vec)` (line 20): 
  - `test_apply_overlay_substitutes_detected_eigenvalues()` (line 66): 
  - `test_apply_overlay_respects_detection_cap()` (line 77): 
  - `test_detect_spikes_uses_tyler_edge_mode()` (line 89): 
  - `test_apply_overlay_with_ewma_shrinker_matches_baseline()` (line 109): 
  - `test_apply_overlay_with_quest_shrinker_matches_baseline()` (line 119): 
  - `test_apply_overlay_with_lw_shrinker_matches_baseline()` (line 129): 
  - `test_apply_overlay_with_oas_shrinker_matches_baseline()` (line 139): 
  - `test_apply_overlay_with_cc_shrinker_matches_baseline()` (line 149): 
  - `test_detect_spikes_strict_gate_filters(monkeypatch)` (line 159): 
  - `test_detect_spikes_handles_missing_stats(monkeypatch)` (line 199): 
  - `test_detect_spikes_preserves_precomputed_stats(monkeypatch)` (line 214): 
  - `test_detect_spikes_soft_gate_selects_top_score(monkeypatch)` (line 232): 
  - `test_detect_spikes_rejects_non_admissible_root(monkeypatch)` (line 270): 
  - `test_detect_spikes_uses_calibrated_delta(tmp_path, monkeypatch)` (line 291): 
  - `test_detect_spikes_rejects_when_calibrated_delta_below_min(tmp_path, monkeypatch)` (line 335): 
  - `test_detect_spikes_records_pre_gate_stats(monkeypatch)` (line 379): 
  - `test_detect_spikes_coarse_candidate_fallback(monkeypatch)` (line 411): 

## fjs.theta_solver
Path: `src/fjs/theta_solver.py`
- Classes:
  - `ThetaSolverParams` (line 12): Closed-form parameters required for the θ root-finding routine.
- Functions:
  - `_normalise_angle(theta)` (line 28): Return θ reduced to the principal interval [0, 2π).
  - `solve_theta_for_t2_zero(lambda_hat, params)` (line 34): Solve for θ such that t₂(λ̂, θ) = 0 for k=2 balanced designs.

## generate_project_state
Path: `tools/generate_project_state.py`
- Functions:
  - `rel_path(path)` (line 55): 
  - `should_skip_dir(relative_dir)` (line 59): 
  - `categorize(path)` (line 73): 
  - `collect_files()` (line 95): 
  - `module_name_from_path(py_path)` (line 128): 
  - `get_py_files()` (line 140): 
  - `parse_symbols(py_files)` (line 157): 
  - `extract_make_targets(makefile)` (line 235): 
  - `main()` (line 262): 

## io.crsp_daily
Path: `src/io/crsp_daily.py`
- Classes:
  - `CrspQueryParams` (line 23): Configuration for the CRSP daily snapshot.
- Functions:
  - `explain_rowcount(sql)` (line 76): Return the planner's estimated rowcount for the provided query.
  - `_clean_snapshot(frame)` (line 97): 
  - `fetch_crsp_daily_snapshot(out_path)` (line 137): Fetch CRSP daily snapshot and persist to parquet.
  - `build_dow_vol_labels(returns)` (line 154): Compute day-of-week and volatility-state labels.
  - `write_labels_parquet(labels, out_path)` (line 202): 

## io.test_wrds_snapshot
Path: `tests/io/test_wrds_snapshot.py`
- Functions:
  - `test_wrds_returns_snapshot_columns()` (line 9): 
  - `test_wrds_labels_snapshot_columns()` (line 20): 

## io.wrds_connect
Path: `src/io/wrds_connect.py`
- Functions:
  - `wrds_conn()` (line 8): 

## list_runs
Path: `tools/list_runs.py`
- Classes:
  - `RunInfo` (line 35): 
- Functions:
  - `_load_json(path)` (line 13): 
  - `_load_detection_total(path)` (line 23): 
  - `_extract_run_info(label, path)` (line 48): 
  - `discover_runs(base_dir)` (line 90): 
  - `format_runs(runs)` (line 107): 
  - `main()` (line 158): 

## make_summary
Path: `tools/make_summary.py`
- Classes:
  - `SummaryArtifacts` (line 20): 
- Functions:
  - `_read_csv(path)` (line 26): 
  - `_normalise(series, value)` (line 35): 
  - `_pick_row(df)` (line 41): 
  - `_pick_dm_row(df)` (line 57): 
  - `_aggregate_row(df)` (line 71): Aggregate matching rows (mean of numeric columns, first of non-numeric).
  - `_aggregate_dm_row(df)` (line 92): 
  - `_nan_median(series)` (line 110): 
  - `_nan_quantile(series, q)` (line 117): 
  - `_count_nonzero(series)` (line 125): 
  - `_concat_if_exists(paths)` (line 132): 
  - `_aggregate_diag_row(df)` (line 143): 
  - `_numeric(series, key)` (line 158): 
  - `_string(series, key, default)` (line 167): 
  - `_load_detail(rc_dir, regime, root_detail)` (line 173): 
  - `_row_for(perf_df, regime, portfolio)` (line 187): 
  - `_criterion_entry(key, label, value, passed, threshold)` (line 199): 
  - `_evaluate_kill_criteria(perf_df, det_df, rc_run, regime)` (line 215): 
  - `summarise_rc_directory(rc_dir)` (line 318): 
  - `_discover_rc_dirs(root, patterns, all_runs, rc_dir)` (line 485): 
  - `_display_path(path)` (line 504): 
  - `write_summaries(rc_dirs)` (line 511): 
  - `parse_args(argv)` (line 560): 
  - `main(argv)` (line 588): 

## manual.merge_calibration_thresholds
Path: `scripts/manual/merge_calibration_thresholds.py`
- Functions:
  - `parse_args()` (line 21): 
  - `_load_json(path)` (line 45): 
  - `_sorted_unique(values)` (line 52): 
  - `main()` (line 56): 

## meta.cache
Path: `src/meta/cache.py`
- Functions:
  - `window_key(manifest, week_list, tickers, replicates)` (line 13): Stable hash identifying a per-window cache entry.
  - `save_window(cache_dir, key, payload)` (line 52): Persist cached per-window statistics.
  - `load_window(cache_dir, key)` (line 77): Load cached per-window statistics if available.

## meta.completeness
Path: `src/meta/completeness.py`
- Classes:
  - `CompletenessResult` (line 38): 
- Functions:
  - `_load_json(path)` (line 9): Read JSON from ``path`` if it exists; otherwise return an empty mapping.
  - `_coerce_int(value)` (line 21): 
  - `_first(mapping, keys)` (line 30): 
  - `_window_stats_from_manifest(manifest)` (line 72): 
  - `evaluate_eval_run(run_dir)` (line 96): Assess completeness for a daily overlay evaluation (rc-lite/rc) run directory.
  - `_locate_payload_dir(base)` (line 154): Return the most likely payload directory (handles tagged weekly outputs).
  - `evaluate_weekly_run(run_dir)` (line 167): Assess completeness for a weekly equity_panel run directory.

## meta.run_meta
Path: `src/meta/run_meta.py`
- Classes:
  - `RunMeta` (line 14): Lightweight metadata summary for a single run.
- Functions:
  - `code_signature(targets)` (line 74): Compute a SHA-256 signature over core de-aliasing code.
  - `_git_sha()` (line 112): Return the short git SHA for the current repository, or 'unknown'.
  - `_sha256_of_file(path)` (line 122): 
  - `_collect_pdf_hashes(directory)` (line 130): 
  - `_load_optional_json(path)` (line 142): 
  - `_count_detections(det_summary_path)` (line 152): Return (detections_total, L_max) from detection_summary.csv if present.
  - `write_run_meta(output_dir)` (line 168): Create a run_meta.json artifact in ``output_dir``.

## meta.runtime
Path: `src/meta/runtime.py`
- Classes:
  - `ExecModeSettings` (line 26): Resolved execution-mode settings shared across runners.
- Functions:
  - `_set_threadpool_limits(max_threads)` (line 34): 
  - `_apply_thread_caps(max_threads)` (line 49): 
  - `resolve_exec_mode(mode)` (line 56): Return the execution-mode settings without mutating global state.
  - `configure_exec_mode(mode)` (line 72): Resolve and apply execution-mode thread caps.
  - `effective_worker_count(settings, requested_workers, cpu_count)` (line 80): Return the worker count respecting the resolved execution mode.
  - `thread_caps_snapshot()` (line 98): Return the current BLAS/OpenMP thread caps for logging.
  - `exec_mode_metadata(settings)` (line 104): Helper to expose execution-mode metadata for run.json payloads.

## pairing
Path: `pairing.py`
- Functions:
  - `_normalize_rows(mat)` (line 8): 
  - `align_spikes(true_vecs, est_vecs)` (line 18): Compute an assignment that pairs estimated spike directions to true ones.

## plot_rc_hist
Path: `tools/plot_rc_hist.py`
- Functions:
  - `_load_series(path, metric, regime)` (line 11): 
  - `plot_histogram(diagnostics_path, metric, out_path)` (line 20): 
  - `main()` (line 49): 

## plotting.utils
Path: `src/plotting/utils.py`
- Functions:
  - `_figures_dir_for_run(run)` (line 19): Return figures directory for a given run under experiments/<run>/figures.
  - `e1_plot_spectrum_with_mp(eigenvalues, mp_edges)` (line 27): E1: Plot spectrum with MP edge and mark outliers.
  - `e2_plot_spike_timeseries(time_index, aliased_series, dealiased_series)` (line 62): E2: Plot aliased vs de-aliased spike time-series.
  - `e3_plot_var_mse(errors_by_method)` (line 101): E3: Single-chart Var-MSE comparison across methods (bar of means).
  - `e4_plot_var_coverage(coverage_errors)` (line 150): E4: VaR(95%) coverage error plot.
  - `s4_plot_guardrails_from_csv(csv_path)` (line 167): S4: Plot guardrail false-positive comparison from a CSV.

## prewhiten
Path: `experiments/prewhiten.py`
- Classes:
  - `PrewhitenTelemetry` (line 26): 
- Functions:
  - `identity_prewhiten_result(returns, factor_cols)` (line 37): 
  - `select_prewhiten_factors(factors, requested)` (line 72): 
  - `_beta_abs_stats(betas)` (line 97): 
  - `compute_prewhiten_telemetry(whitening)` (line 113): 
  - `apply_prewhitening(returns)` (line 140): 
  - `write_prewhiten_diagnostics(out_dir, whitening, telemetry)` (line 169): 

## prewhiten_effect
Path: `tools/prewhiten_effect.py`
- Classes:
  - `RunSummary` (line 53): 
- Functions:
  - `_read_csv_rows(path)` (line 13): 
  - `_scalar(value)` (line 26): 
  - `_mode_from_resolved(run_dir)` (line 33): 
  - `_portfolio_value(rows, portfolio, column)` (line 62): 
  - `_sign_p_value(path, portfolio)` (line 73): 
  - `_load_run_summary(run_dir)` (line 85): 
  - `_build_effect_rows(off, on, label_off, label_on)` (line 117): 
  - `parse_args(argv)` (line 150): 
  - `main(argv)` (line 172): 

## reduce_calibration
Path: `tools/reduce_calibration.py`
- Functions:
  - `parse_args(argv)` (line 14): 
  - `main(argv)` (line 58): 

## report.gather
Path: `src/report/gather.py`
- Functions:
  - `load_run(path)` (line 28): Load core artifacts for a single run directory.
  - `find_runs(root, pattern)` (line 72): Discover run directories, preferring tagged folders.
  - `_extract_detection(summary_df)` (line 98): 
  - `_extract_edge_stats(summary_df)` (line 104): 
  - `_dm_values(de_row, estimator)` (line 119): 
  - `_dm_values_qlike(de_row, estimator)` (line 130): 
  - `_ci_bounds(de_row, estimator)` (line 141): 
  - `collect_estimator_panel(run_paths)` (line 164): Combine estimator diagnostics across runs into a single table.

## report.plots
Path: `src/report/plots.py`
- Functions:
  - `_single_run_tag(df)` (line 23): 
  - `_ensure_dir(path)` (line 29): 
  - `plot_dm_bars(df)` (line 33): 
  - `plot_edge_margin_hist(df)` (line 73): 
  - `plot_detection_rate(df)` (line 102): 
  - `plot_alignment_angles(df)` (line 127): 
  - `plot_ablation_heatmap(df)` (line 157): 

## report.tables
Path: `src/report/tables.py`
- Functions:
  - `_single_run_tag(df)` (line 19): 
  - `_ensure_dir(path)` (line 25): 
  - `_format_float(value)` (line 29): 
  - `_write_markdown(df, path)` (line 37): 
  - `_find_strategy_row(group, candidates)` (line 51): 
  - `table_estimators_panel(df)` (line 62): Create estimator panel comparison tables and return paths to CSV, Markdown, and LaTeX outputs.
  - `table_rejections(df)` (line 191): Generate a rejection reason summary table.
  - `table_ablation(df)` (line 236): Summarise ablation grids when available.

## run_monitor
Path: `tools/run_monitor.py`
- Classes:
  - `MetricSample` (line 44): 
- Functions:
  - `_now_iso()` (line 32): 
  - `_safe_json(line)` (line 36): 
  - `_aggregate_process_metrics(proc)` (line 56): 
  - `_io_counters(proc)` (line 83): 
  - `_monitor_loop(proc, interval, metrics_path, progress_queue, stop_event, hostname, samples)` (line 107): 
  - `_summarise(samples)` (line 215): 
  - `main()` (line 239): 

## shard_grid
Path: `tools/shard_grid.py`
- Functions:
  - `parse_args(argv)` (line 14): 
  - `_shard_jobs(jobs, shard_count, strategy)` (line 62): 
  - `main(argv)` (line 79): 

## summarize_rc_sanity
Path: `tools/summarize_rc_sanity.py`
- Functions:
  - `_delta_mse(metrics, portfolio)` (line 22): 
  - `_effect_label(delta_ew, delta_mv)` (line 33): 
  - `_load_daily_payload(path, label)` (line 48): Best-effort loader for daily diagnostics/metrics.
  - `_load_weekly_payload(path)` (line 103): Best-effort loader for weekly summary + detection.
  - `_merge_completeness(entry, comp)` (line 148): 
  - `_build_daily_entry(label, path)` (line 165): 
  - `_build_weekly_entry(label, path)` (line 178): 
  - `_aggregate_entries(entries)` (line 187): 
  - `main()` (line 207): 

## summarize_run
Path: `tools/summarize_run.py`
- Functions:
  - `_read_json(path)` (line 15): 
  - `_safe_read_csv(path)` (line 25): 
  - `_fmt_float(x)` (line 34): 
  - `summarize_run(output_dir)` (line 40): 
  - `main()` (line 303): 

## summarize_weekly_diagnostics
Path: `tools/summarize_weekly_diagnostics.py`
- Functions:
  - `_format_skip_summary(df, top_k)` (line 12): 
  - `_render_stat_table(df, columns)` (line 28): 
  - `_guardrail_totals(df)` (line 42): 
  - `summarize(input_path, output_path, top_k)` (line 52): 
  - `main()` (line 94): 

## synthetic.calibrate_thresholds
Path: `experiments/synthetic/calibrate_thresholds.py`
- Functions:
  - `_parse_float_list(values)` (line 29): 
  - `parse_args(argv)` (line 41): 
  - `build_planned_jobs(p_assets_list, n_groups_list, replicates_list, delta_abs_grid)` (line 181): Return the Cartesian product of sweep dimensions (excluding edge modes).
  - `_parse_bins(specs)` (line 205): 
  - `_assign_bin(value, bins)` (line 229): 
  - `_maybe_plot(entries, alpha, path)` (line 236): 
  - `_cell_identifier(p_assets, n_groups, replicates, delta_abs, edge_mode)` (line 284): 
  - `_load_shard_jobs(manifest_path, shard_id)` (line 288): 
  - `_build_cell_records(config, result, edge_mode)` (line 318): 
  - `_write_cell_payload(path, payload)` (line 355): 
  - `_load_cell_payloads(cells_dir)` (line 362): 
  - `_collect_cell_records(cell_payloads)` (line 374): 
  - `_build_threshold_map(entries, replicate_bins, asset_bins, alpha)` (line 385): 
  - `_build_defaults_payload(thresholds_map)` (line 454): 
  - `main(argv)` (line 519): 
  - `_git_sha()` (line 766): 
  - `_blas_info()` (line 775): 
  - `_instance_metadata()` (line 788): 

## synthetic.calibration
Path: `src/synthetic/calibration.py`
- Classes:
  - `CalibrationConfig` (line 26): Configuration for synthetic calibration of overlay thresholds.
  - `ThresholdEntry` (line 48): 
  - `GridStat` (line 70): 
  - `CalibrationResult` (line 94): 
- Functions:
  - `_simulate_panel(config, rng)` (line 122): 
  - `_select_entry(candidates, alpha)` (line 145): 
  - `calibrate_thresholds(config)` (line 163): 
  - `write_thresholds(result, path)` (line 244): 
  - `_run_seed_batches()` (line 252): 
  - `_seed_batch_worker(payload)` (line 299): 

## synthetic.harness_utils
Path: `experiments/synthetic/harness_utils.py`
- Classes:
  - `HarnessConfig` (line 27): Configuration for synthetic null/power harness simulations.
  - `ScoreResult` (line 53): Container for per-trial spectral scores.
  - `SimulatedScores` (line 79): Structured return for score simulations.
  - `EnergyFloorSelection` (line 224): 
- Functions:
  - `_compute_scatter(observations, edge_mode)` (line 91): 
  - `_score_trial(observations, edge_mode)` (line 106): 
  - `_run_single_mu(config, mu)` (line 121): 
  - `simulate_scores(config, mu_values)` (line 163): Simulate score distributions for the supplied spike strengths.
  - `roc_table(null_scores, power_scores)` (line 181): Return a ROC-style table (FPR vs power) per edge mode and spike.
  - `select_energy_floor(null_scores, power_scores)` (line 241): Select an energy floor that satisfies the FPR cap while maximising power.
  - `write_run_metadata(path)` (line 300): 

## synthetic.nested_killtest
Path: `experiments/synthetic/nested_killtest.py`
- Classes:
  - `TrialResult` (line 64): 
- Functions:
  - `load_config(path)` (line 77): 
  - `simulate_nested_panel(rng)` (line 88): Return (observations, year_labels, week_labels).
  - `_edge_scale(observations, edge_mode, edge_huber_c)` (line 133): 
  - `run_trials(config)` (line 172): 
  - `summarise_results(results)` (line 367): 
  - `write_summary_markdown(summary_df, out_path)` (line 394): 
  - `main(argv)` (line 406): 

## synthetic.null
Path: `experiments/synthetic/null.py`
- Functions:
  - `parse_args(argv)` (line 32): 
  - `_git_sha()` (line 63): 
  - `_build_fpr_curve(scores)` (line 70): 
  - `_plot_fpr_curve(curve, path)` (line 89): 
  - `main(argv)` (line 104): 

## synthetic.power
Path: `experiments/synthetic/power.py`
- Functions:
  - `parse_args(argv)` (line 40): 
  - `_git_sha()` (line 95): 
  - `_load_null_scores(path)` (line 102): 
  - `_plot_roc(roc, mu_values, selection, path)` (line 115): 
  - `_save_defaults(path)` (line 145): 
  - `main(argv)` (line 180): 

## synthetic.power_null
Path: `experiments/synthetic/power_null.py`
- Classes:
  - `TrialResult` (line 200): 
- Functions:
  - `_resolve_delta_grid(delta_grid)` (line 76): 
  - `_normalise_for_meta(value)` (line 83): 
  - `_normalise_panel_specs(panel_specs)` (line 95): 
  - `calibration_cache_meta()` (line 107): 
  - `load_calibration_cache(path, meta, dependencies)` (line 130): 
  - `write_calibration_cache(path, payload, meta)` (line 161): 
  - `_edge_scale_for_mode(y, mode, huber_c)` (line 169): 
  - `_detections_for_mode(y, groups)` (line 210): 
  - `_simulate_null(config)` (line 255): 
  - `_simulate_power(config)` (line 267): 
  - `run_trials()` (line 297): 
  - `calibrate_delta_thresholds()` (line 370): Estimate minimal delta_frac values achieving target null FPR for each (p, T).
  - `summarise_results(results)` (line 481): 
  - `plot_fpr_heatmap(summary, out_path)` (line 507): 
  - `plot_power_curves(summary, out_path)` (line 527): 
  - `parse_args()` (line 550): 
  - `main()` (line 644): 

## synthetic.test_calibration
Path: `tests/synthetic/test_calibration.py`
- Functions:
  - `test_calibrate_thresholds_controls_fpr(tmp_path)` (line 12): 
  - `test_edge_delta_thresholds_real_run()` (line 42): 
  - `test_calibration_deterministic_small_grid()` (line 64): 
  - `test_calibration_resume_equivalence(tmp_path, monkeypatch)` (line 85): 

## synthetic.test_harness_utils
Path: `tests/synthetic/test_harness_utils.py`
- Functions:
  - `_small_config(edge_modes)` (line 11): 
  - `test_simulate_scores_increases_with_signal()` (line 24): 
  - `test_select_energy_floor_respects_target_fpr()` (line 34): 
  - `test_roc_table_emits_entries_for_each_mu()` (line 48): 

## synthetic.threshold_eval
Path: `src/synthetic/threshold_eval.py`
- Classes:
  - `DetectionArrays` (line 15): 
- Functions:
  - `_extract_detection_arrays(detections)` (line 28): 
  - `_evaluate_delta_grid(arrays)` (line 99): 
  - `evaluate_threshold_grid(observations, groups)` (line 127): 

## synthetic_oneway.run
Path: `experiments/synthetic_oneway/run.py`
- Functions:
  - `load_config(path)` (line 69): Load experiment configuration from YAML, falling back to defaults.
  - `ensure_dir(path)` (line 83): Create ``path`` and parents if they do not yet exist.
  - `simulate_panel(rng)` (line 89): Simulate a balanced MANOVA panel with a single spike.
  - `simulate_multi_spike(rng)` (line 130): Simulate a panel with multiple planted spikes.
  - `mp_upper_edge(noise_variance, n_assets, n_groups)` (line 170): Return the Marčenko–Pastur upper edge for the supplied regime.
  - `histogram_s1(eigenvalues, edge, out_dir)` (line 177): Save the S1 histogram visualising the empirical spectrum.
  - `bias_table_s3(df, out_dir)` (line 197): Persist the S3 bias summary to disk.
  - `s2_vector_alignment(config, rng)` (line 203): Evaluate alignment between the leading eigvector and the planted spike.
  - `summary_to_json(summary, out_dir)` (line 252): Write a JSON summary of the synthetic experiments.
  - `s1_monte_carlo(config, rng)` (line 259): Run the S1 Monte Carlo sweep and return summary statistics.
  - `s3_bias(config, rng)` (line 295): Evaluate aliased versus de-aliased bias across spike strengths.
  - `s4_guardrail_analysis(config, rng)` (line 373): Compare false-positive rates under default versus lax guardrails.
  - `s5_multi_spike_bias(config, rng)` (line 489): Assess bias reduction in a multi-spike setting.
  - `plot_bias_timeseries(prefixes, aliased, dealiased, spike, output_dir)` (line 695): Plot a diagnostic bias timeseries for an individual spike size.
  - `run_experiment(config_path)` (line 716): Execute the S1/S3 synthetic experiments.
  - `main()` (line 764): Entry point for CLI execution.

## test_aggregate_runs
Path: `tests/test_aggregate_runs.py`
- Functions:
  - `_write_metrics(path)` (line 10): 
  - `_write_summary(path)` (line 19): 
  - `test_aggregate_runs_cli(tmp_path)` (line 27): 

## test_alignment_diag
Path: `tests/test_alignment_diag.py`
- Functions:
  - `test_alignment_angle_boundaries()` (line 8): 
  - `test_alignment_angle_handles_full_rank()` (line 17): 

## test_balanced
Path: `tests/test_balanced.py`
- Functions:
  - `_balanced_panel()` (line 16): 
  - `test_group_means_returns_expected_statistics()` (line 29): 
  - `test_mean_squares_matrices_and_identities()` (line 43): 
  - `test_mean_squares_rejects_unbalanced_design()` (line 71): 
  - `test_compute_balanced_weights_is_stub()` (line 82): 

## test_balanced_sigma_a
Path: `tests/test_balanced_sigma_a.py`
- Functions:
  - `test_sigma_of_a_balanced_design_identity()` (line 8): 

## test_balanced_weeks
Path: `tests/test_balanced_weeks.py`
- Functions:
  - `test_balance_weeks_drops_partial_periods()` (line 10): 

## test_cache
Path: `tests/test_cache.py`
- Functions:
  - `_write_returns_csv(path)` (line 16): 
  - `test_window_cache_resume(monkeypatch, tmp_path)` (line 36): 

## test_cache_switch_estimator
Path: `tests/test_cache_switch_estimator.py`
- Functions:
  - `test_window_cache_key_changes_with_estimator(tmp_path)` (line 11): 

## test_calibrate_defaults
Path: `tests/test_calibrate_defaults.py`
- Functions:
  - `test_build_defaults_payload_filters_and_formats(tmp_path)` (line 9): 

## test_calibration_lookup
Path: `tests/test_calibration_lookup.py`
- Functions:
  - `test_lookup_calibrated_delta_nested_tyler()` (line 4): 
  - `test_lookup_calibrated_delta_nested_huber()` (line 12): 

## test_data_registry
Path: `tests/test_data_registry.py`
- Functions:
  - `_write_registry(path, dataset_key, sha)` (line 12): 
  - `_sha256(path)` (line 29): 
  - `test_assert_registered_dataset_valid(monkeypatch, tmp_path)` (line 37): 
  - `test_assert_registered_dataset_hash_mismatch(monkeypatch, tmp_path)` (line 58): 

## test_dealias
Path: `tests/test_dealias.py`
- Functions:
  - `_simulate_one_way(rng)` (line 17): 
  - `test_dealiasing_result_structure()` (line 49): 
  - `test_dealias_covariance_uses_detection_vectors()` (line 55): 
  - `test_dealias_covariance_accepts_target_spectrum()` (line 71): 
  - `test_dealias_search_detects_sigma1_spike()` (line 81): 
  - `test_t_vector_acceptance_consistency_toy_spike()` (line 104): 
  - `test_relative_delta_enables_detection_when_absolute_blocks()` (line 158): 
  - `test_equity_toy_detection_with_delta_frac()` (line 184): 
  - `test_detections_include_diagnostics_fields()` (line 210): 
  - `test_signed_a_grid_no_crash()` (line 245): 
  - `test_dealias_search_limits_sigma2_false_positives()` (line 275): 
  - `test_dealias_search_has_low_false_positive_rate()` (line 298): 
  - `test_dealias_search_isotropic_trials_under_one_percent()` (line 329): 
  - `test_cs_drop_top_frac_influences_threshold_or_detections()` (line 352): 
  - `test_dealias_search_stability_consistent_across_eta()` (line 394): 

## test_dealias_guardrails
Path: `tests/test_dealias_guardrails.py`
- Functions:
  - `_simulate_balanced_panel(rng)` (line 15): 
  - `test_dealias_rejects_isotropic_panels()` (line 34): 
  - `test_lax_guardrails_raise_fpr()` (line 64): 
  - `test_dealias_detections_are_angularly_stable()` (line 108): 

## test_dealias_search
Path: `tests/test_dealias_search.py`
- Functions:
  - `_make_single_spike_panel(rng)` (line 17): 
  - `test_single_spike_mu_hat_within_standard_error()` (line 37): 
  - `test_permuted_group_labels_yield_no_detections()` (line 65): 
  - `test_decisions_stable_within_eta_band()` (line 92): 
  - `test_wrds_window_yields_detection_with_relaxed_leakage()` (line 150): 

## test_diagnostics
Path: `tests/test_diagnostics.py`
- Functions:
  - `_write_returns_csv(path)` (line 19): 
  - `test_window_artifacts_expose_edge_margin_and_admissible_root(tmp_path)` (line 39): 

## test_dm
Path: `tests/test_dm.py`
- Functions:
  - `test_dm_test_detects_mean_difference()` (line 11): 
  - `test_dm_test_handles_identical_or_missing_losses()` (line 27): 

## test_e2_timeseries
Path: `tests/test_e2_timeseries.py`
- Functions:
  - `_make_prices_csv(tmp_path, weeks, assets)` (line 13): 
  - `test_e2_spike_timeseries_written(tmp_path)` (line 29): 

## test_equity_ablations
Path: `tests/test_equity_ablations.py`
- Functions:
  - `_make_prices_csv(tmp_path, weeks, assets)` (line 13): 
  - `test_equity_ablation_emits_summary(tmp_path)` (line 34): 

## test_equity_prewhiten
Path: `tests/test_equity_prewhiten.py`
- Functions:
  - `_write_returns_csv(tmp_path)` (line 14): 
  - `_write_factor_csv(tmp_path)` (line 34): 
  - `test_run_experiment_emits_prewhiten_columns(tmp_path)` (line 47): 

## test_eval_missing_solver
Path: `tests/test_eval_missing_solver.py`
- Functions:
  - `_raise_missing()` (line 10): 
  - `test_min_variance_weights_raises_when_cvxpy_missing(monkeypatch)` (line 14): 
  - `test_min_variance_weights_skip_missing(monkeypatch)` (line 21): 

## test_evaluation_checks
Path: `tests/test_evaluation_checks.py`
- Functions:
  - `test_check_dealiased_applied_raises_on_identical_forecasts()` (line 11): 
  - `test_check_dealiased_applied_passes_when_different()` (line 27): 

## test_evaluation_metrics
Path: `tests/test_evaluation_metrics.py`
- Functions:
  - `test_sign_test_pvalue_strong_asymmetry()` (line 19): 
  - `test_block_bootstrap_ci_median_contains_sample_median()` (line 26): 
  - `test_iqr_matches_percentile_difference()` (line 40): 
  - `test_build_metrics_summary_dealiased_vs_baselines()` (line 45): 

## test_factor_baselines
Path: `tests/test_factor_baselines.py`
- Functions:
  - `test_observed_factor_covariance_matches_population()` (line 9): 
  - `test_poet_lite_covariance_returns_valid_matrix()` (line 34): 

## test_factor_cov
Path: `tests/test_factor_cov.py`
- Functions:
  - `test_factor_covariance_matches_theoretical_when_noise_zero()` (line 12): 
  - `test_factor_covariance_with_industry_and_missing_returns()` (line 35): 

## test_gating
Path: `tests/test_gating.py`
- Functions:
  - `test_count_isolated_outliers_zero_when_missing_isolated()` (line 11): 
  - `test_select_top_k_prefers_high_score_and_edge_margin()` (line 19): 
  - `test_lookup_calibrated_delta_reads_json(tmp_path)` (line 50): 

## test_gpt_bundle
Path: `tests/test_gpt_bundle.py`
- Functions:
  - `test_makefile_has_gpt_bundle_target_and_inputs()` (line 19): 

## test_metrics_qlike
Path: `tests/test_metrics_qlike.py`
- Functions:
  - `test_qlike_matches_manual_computation()` (line 8): 

## test_minvar_regularized
Path: `tests/test_minvar_regularized.py`
- Functions:
  - `test_minvar_ridge_box_respects_box_and_sum_constraints()` (line 11): 
  - `test_minvar_ridge_box_matches_objective_and_improves_conditioning()` (line 30): 
  - `test_turnover_and_turnover_cost_application()` (line 52): 
  - `test_minvar_ridge_box_enforces_narrow_box()` (line 70): 
  - `test_minvar_ridge_box_handles_near_singular_covariance()` (line 84): 
  - `test_minvar_memoization_matches_plain_solver()` (line 97): 

## test_mp
Path: `tests/test_mp.py`
- Functions:
  - `micro_mp_params()` (line 25): 
  - `test_z_of_m_agrees_with_reference(micro_mp_params)` (line 56): 
  - `test_mp_edge_below_outlier(micro_mp_params)` (line 69): 
  - `test_admissible_root_matches_reference(micro_mp_params)` (line 77): 
  - `test_mp_edge_uses_Cs_in_denominator()` (line 89): 
  - `test_t_vec_monotonicity(micro_mp_params)` (line 106): 
  - `test_marchenko_pastur_edges_is_stub()` (line 135): 
  - `test_marchenko_pastur_pdf_is_stub()` (line 141): 
  - `test_mp_edge_cache_parity(tmp_path, micro_mp_params)` (line 148): 

## test_mp_edge_and_root
Path: `tests/test_mp_edge_and_root.py`
- Functions:
  - `_balanced_params()` (line 19): 
  - `test_derivatives_match_finite_differences()` (line 28): 
  - `test_round_trip_lambda_to_m_to_lambda()` (line 51): 
  - `test_concavity_at_edge()` (line 65): 

## test_nested_balanced
Path: `tests/test_nested_balanced.py`
- Functions:
  - `_generate_nested_sample(I, J, R, p)` (line 11): 
  - `test_mean_squares_nested_balanced()` (line 47): 
  - `test_nested_dealias_smoke_run()` (line 86): 

## test_nested_smoke
Path: `tests/test_nested_smoke.py`
- Functions:
  - `_synthetic_nested_blocks()` (line 14): Generate a small balanced Year⊃Week panel with a Σ₁ spike.
  - `test_nested_smoke_detection_positive_stability()` (line 51): 

## test_pairing
Path: `tests/test_pairing.py`
- Functions:
  - `test_pairing_alignment_improves_median_bias(tmp_path)` (line 13): 

## test_pipeline_smoke
Path: `tests/test_pipeline_smoke.py`
- Functions:
  - `test_core_types_are_accessible()` (line 45): 
  - `test_core_functions_are_callable()` (line 53): 
  - `test_experiment_entry_points_are_callable()` (line 76): 
  - `test_experiment_configs_load()` (line 82): 
  - `test_single_equity_window_smoke(tmp_path)` (line 92): 
  - `test_weekly_components_identity_and_detection_gain()` (line 150): 
  - `test_relative_delta_and_signed_a_integration()` (line 219): 
  - `test_rolling_synthetic_oos_gain()` (line 249): 

## test_plotting_utils
Path: `tests/test_plotting_utils.py`
- Functions:
  - `test_e1_e2_e3_e4_and_s4_create_pdfs(tmp_path, run_name)` (line 32): 

## test_portfolios_missing_solver
Path: `tests/test_portfolios_missing_solver.py`
- Functions:
  - `_raise_missing()` (line 9): 
  - `test_minimum_variance_raises_when_solver_missing(monkeypatch)` (line 13): 
  - `test_optimize_portfolio_raises_by_default(monkeypatch)` (line 21): 
  - `test_optimize_portfolio_skip_flag_marks_skipped(monkeypatch)` (line 29): 
  - `test_force_missing_env_triggers_skip(monkeypatch)` (line 43): 

## test_power_null
Path: `tests/test_power_null.py`
- Functions:
  - `test_power_null_summary_behaves()` (line 23): 
  - `test_calibrate_delta_thresholds_returns_lookup()` (line 62): 
  - `test_calibrate_delta_thresholds_multi_panel_fpr_cap()` (line 89): 
  - `test_calibration_cache_roundtrip(tmp_path)` (line 114): 
  - `test_calibration_cache_force_recomputes(tmp_path)` (line 160): 

## test_report_gather
Path: `tests/test_report_gather.py`
- Functions:
  - `test_load_run_frames()` (line 15): 
  - `test_find_runs_prefers_tagged(tmp_path)` (line 24): 
  - `test_collect_estimator_panel()` (line 37): 

## test_report_plots
Path: `tests/test_report_plots.py`
- Functions:
  - `test_plot_dm_and_detection(tmp_path)` (line 21): 
  - `test_plot_edge_margin(tmp_path)` (line 29): 
  - `test_plot_ablation_heatmap(tmp_path)` (line 40): 

## test_report_tables
Path: `tests/test_report_tables.py`
- Functions:
  - `test_table_estimators(tmp_path)` (line 16): 
  - `test_table_rejections(tmp_path)` (line 26): 
  - `test_table_ablation(tmp_path)` (line 41): 

## test_robust_edge
Path: `tests/test_robust_edge.py`
- Functions:
  - `test_tyler_scatter_returns_psd_matrix()` (line 8): 
  - `test_edge_from_scatter_monotone_in_scale()` (line 20): 

## test_run_meta_module
Path: `tests/test_run_meta_module.py`
- Functions:
  - `test_write_run_meta_creates_file_with_expected_fields(tmp_path)` (line 15): 

## test_s5_pairing_alignment
Path: `tests/test_s5_pairing_alignment.py`
- Functions:
  - `test_s5_alignment_pairing_comparison(tmp_path)` (line 13): 

## test_sharadar_pipeline_smoke
Path: `tests/test_sharadar_pipeline_smoke.py`
- Functions:
  - `test_sharadar_fetch_and_balance_smoke(tmp_path)` (line 18): 

## test_shrinkage
Path: `tests/test_shrinkage.py`
- Functions:
  - `test_oas_covariance_returns_psd()` (line 14): 
  - `test_constant_correlation_shrinkage_reduces_off_diagonal_weight()` (line 26): 
  - `test_shrinkers_warn_on_nonfinite_and_remain_psd(caplog)` (line 45): 
  - `test_winsorize_clips_extremes()` (line 65): 
  - `test_huberize_limits_outliers()` (line 74): 
  - `test_tyler_shrink_covariance_is_positive_definite()` (line 81): 

## test_sweep_cli
Path: `tests/test_sweep_cli.py`
- Functions:
  - `test_sweep_dry_run(tmp_path)` (line 8): 

## test_theta_solver
Path: `tests/test_theta_solver.py`
- Functions:
  - `_synthetic_oneway_dataset()` (line 13): 
  - `test_theta_solver_brackets_root(monkeypatch)` (line 36): 
  - `test_theta_solver_fallback_to_grid(monkeypatch)` (line 85): 
  - `test_theta_solver_logs_solver_flag()` (line 117): 

## test_threshold_eval
Path: `tests/test_threshold_eval.py`
- Functions:
  - `test_threshold_eval_matches_detect_spikes()` (line 11): 

## test_var_backtests
Path: `tests/test_var_backtests.py`
- Functions:
  - `test_kupiec_pof_rates_relative_misspecification()` (line 12): 
  - `test_christoffersen_independence_flags_clustering()` (line 21): 
  - `test_expected_shortfall_test_reacts_to_bias()` (line 30): 

## test_weekly_cov_identity
Path: `tests/test_weekly_cov_identity.py`
- Functions:
  - `test_weekly_covariance_identity()` (line 7): 

## tools.test_make_summary
Path: `tests/tools/test_make_summary.py`
- Functions:
  - `_copy_sample_rc(tmp_path)` (line 12): 
  - `test_summarise_rc_directory(tmp_path)` (line 21): 

## tools.test_summarize_rc_sanity
Path: `tests/tools/test_summarize_rc_sanity.py`
- Functions:
  - `_write_daily_run(run_dir)` (line 16): 
  - `test_eval_completeness_flags_partial_dir(tmp_path)` (line 52): 
  - `test_summarizer_marks_missing_sections_and_excludes_incomplete(tmp_path)` (line 64): 

## update_registry
Path: `tools/update_registry.py`
- Functions:
  - `compute_sha256(path)` (line 15): 
  - `summarise_returns(path)` (line 23): 
  - `load_registry(path)` (line 33): 
  - `main()` (line 43): 

## utils.credentials
Path: `src/utils/credentials.py`
- Functions:
  - `_kc(service, account)` (line 3): 
  - `wrds_user()` (line 13): 
  - `wrds_password(user)` (line 16): 
  - `wrds_creds()` (line 20): 

## verify_dataset
Path: `tools/verify_dataset.py`
- Functions:
  - `_sha256(path)` (line 11): 
  - `_normalise_key(path)` (line 19): 
  - `_load_registry(path)` (line 27): 
  - `verify_dataset(dataset_path, registry_path)` (line 38): 
  - `main()` (line 64): 