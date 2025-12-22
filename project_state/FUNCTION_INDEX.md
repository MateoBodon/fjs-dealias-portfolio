---
generated: 2025-12-22T21:06:25Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
---


# Function Index


Source: `project_state/_generated/symbol_index.json` (AST-derived; src/, experiments/, tools/ only).

## `experiments/__init__.py`
- module: `experiments`
- classes: none
- functions: none

## `experiments/ablate/__init__.py`
- module: `experiments.ablate`
- classes: none
- functions: none

## `experiments/ablate/run.py`
- module: `experiments.ablate.run`
- classes:
  - `PanelSpec` @ L58
- functions:
  - `_load_yaml(path)` @ L66
  - `_coerce_bool(value)` @ L73
  - `_coerce_value(key, value)` @ L87
  - `_normalise_defaults(defaults)` @ L99
  - `_normalise_combo(combo)` @ L106
  - `_combo_identifier(params)` @ L113
  - `_is_default_combo(combo, defaults, keys)` @ L129
  - `_ensure_dir(path)` @ L150
  - `_load_panels(section)` @ L154
  - `_extract_perf(perf_df, regime, portfolio)` @ L174
  - `_extract_detection(det_df, regime)` @ L183
  - `_safe_get(series, key)` @ L191
  - `run_ablation(config_path, *, force=False, limit=None, calm_window_sample=None, crisis_window_top_k=None)` @ L200
  - `parse_args(argv=None)` @ L440
  - `main(argv=None)` @ L474

## `experiments/daily/__init__.py`
- module: `experiments.daily`
- classes: none
- functions: none

## `experiments/daily/config.py`
- module: `experiments.daily.config`
- classes:
  - `DailyDesign` @ L9 — Configuration defaults for a replicated daily experiment.
- functions: none

## `experiments/daily/grouping.py`
- module: `experiments.daily.grouping`
- classes:
  - `GroupingError` @ L16 (bases: RuntimeError) — Raised when a sliding window cannot be balanced for replicates.
- functions:
  - `_ensure_datetime_index(frame)` @ L20
  - `group_by_week(frame, *, replicates=5)` @ L25 — Balance a window by complete weeks (default: Monday-aligned business weeks).
  - `group_by_day_of_week(frame, *, min_weeks=3)` @ L47 — Balance a window by Day-of-Week replicates across complete weeks.
  - `_vol_state_codes(frame, vol_proxy, *, calm_threshold, crisis_threshold)` @ L95
  - `group_by_vol_state(frame, *, vol_proxy, calm_threshold, crisis_threshold, min_replicates=4)` @ L122 — Balance a window by volatility-state buckets (calm/mid/crisis).
  - `group_by_dow_vol(frame, *, vol_proxy, calm_threshold, crisis_threshold, min_replicates)` @ L175 — Balance windows by crossing day-of-week with volatility states.
  - `group_by_dow_month(frame, *, min_replicates)` @ L225 — Balance windows by crossing day-of-week with calendar month.

## `experiments/daily/run.py`
- module: `experiments.daily.run`
- classes: none
- functions:
  - `_detect_forward_override(argv, flag)` @ L14
  - `parse_args(argv=None)` @ L28
  - `_default_out(design, rc_date)` @ L78
  - `main(argv=None)` @ L84

## `experiments/equity_panel/__init__.py`
- module: `experiments.equity_panel`
- classes: none
- functions: none

## `experiments/equity_panel/reasons.py`
- module: `experiments.equity_panel.reasons`
- classes:
  - `SkipReasonPrimary` @ L8 (bases: str, Enum)
  - `SkipAttribution` @ L90
- functions:
  - `collect_unknown_guards(diag_local)` @ L71 — Return positive-count guard keys not covered by the canonical set.
  - `normalise_diag_counts(diag_local)` @ L98 — Project raw diagnostics to a stable guardrail count dictionary.
  - `infer_primary_reason(diag_local, *, calibration_missing, isolated_spikes, calibration_detail=None)` @ L112 — Map guardrail diagnostics to a stable primary skip reason.

## `experiments/equity_panel/run.py`
- module: `experiments.equity_panel.run`
- classes:
  - `PreparedWindowStats` @ L161
- functions:
  - `_parse_box_bounds(bounds)` @ L138 — Normalise min-variance box bounds into a (lo, hi) tuple.
  - `_load_prepared_from_cache(cached_stats, design_mode, y_fit_raw, code_signature_hash, expected_nested_replicates)` @ L176
  - `_compute_oneway_prepared(fit_blocks, y_fit_raw, replicates, code_signature_hash)` @ L279
  - `_compute_nested_prepared(fit_blocks, y_fit_raw, expected_reps, code_signature_hash)` @ L348
  - `_compute_grouped_design_prepared(y_fit_raw, group_labels, *, design_mode, code_signature_hash)` @ L644 — Compute MANOVA statistics for alternate balanced daily groupings.
  - `_prepare_window_stats(design_mode, fit_blocks, replicates, *, cached_stats=None, nested_replicates=None)` @ L718
  - `_infer_skip_reason(diag_local, *, calibration_missing, calibration_detail=None, isolated_spikes)` @ L748 — Best-effort categorisation for windows with no accepted detections.
  - `load_config(path)` @ L764 — Load experiment configuration, falling back to defaults.
  - `_generate_synthetic_prices(path)` @ L787 — Create a synthetic price panel for quick smoke testing.
  - `_mp_edges(noise_variance, n_assets, n_samples)` @ L812 — Return approximate Marčenko–Pastur bulk edges.
  - `_prepare_data(config)` @ L824 — Load daily returns from returns CSV or derive from prices CSV.
  - `_apply_preprocessing(daily_returns, *, winsorize_q, huber_c)` @ L849 — Apply optional robustness preprocessing to daily returns.
  - `_run_param_ablation(daily_returns, output_dir, *, partial_week_policy, target_component, base_delta, base_delta_frac, base_eps, base_eta, signed_a, off_component_leak_cap, energy_min_abs, oneway_a_solver, preprocess_flags=None, grid_overrides=None, use_tvector=True)` @ L868 — Grid sweep over detection parameters; emit CSV and heatmaps (E5).
  - `_load_or_build_balanced_panel(daily_returns, *, days_per_week, partial_week_policy, output_dir, precompute_panel, preprocess_flags=None)` @ L1108 — Load a cached balanced panel or build a fresh one from daily returns.
  - `_run_single_period(daily_returns, *, start, end, output_dir, window_weeks, horizon_weeks, max_windows, delta, delta_frac, eps, stability_eta, signed_a, target_component, partial_week_policy, precompute_panel, cache_dir, resume_cache, cs_drop_top_frac, cs_sensitivity_frac, off_component_leak_cap, sigma_ablation, label, crisis_label=None, design_mode, nested_replicates, oneway_a_solver, estimator, progress=True, a_grid=120, energy_min_abs=None, factor_returns=None, prewhiten_meta=None, minvar_ridge=0.0001, minvar_box=(0.0, 0.1), turnover_cost_bps=5.0, minvar_condition_cap=1000000000.0, preprocess_flags=None, gating=None, alignment_top_p=3, edge_mode='scm', edge_huber_c=1.5, use_tvector=True, diagnostics=None)` @ L1174 — Execute the rolling evaluation for a single date range.
  - `_run_sigma_ablation(daily_returns, output_dir, cs_drop_top_frac, delta, delta_frac, eps, stability_eta, signed_a, target_component)` @ L3279 — Evaluate Cs perturbations and persist sensitivity diagnostics.
  - `run_experiment(config_path=None, *, output_dir_override=None, sigma_ablation=False, crisis=None, delta_frac_override=None, signed_a_override=None, target_component_override=None, design_override=None, nested_replicates_override=None, oneway_a_solver_override=None, cs_drop_top_frac_override=None, progress_override=None, eps_override=None, a_grid_override=None, ablations=None, eta_override=None, window_weeks_override=None, horizon_weeks_override=None, max_windows_override=None, energy_min_abs_override=None, partial_week_policy=None, precompute_panel=False, cache_dir_override=None, resume_cache=False, estimator_override=None, winsorize_q_override=None, huber_c_override=None, factor_csv_override=None, prewhiten_override=None, use_factor_prewhiten_override=None, minvar_ridge_override=None, minvar_box_override=None, turnover_cost_override=None, minvar_condition_cap_override=None, edge_mode_override=None, edge_huber_c_override=None, gating_mode_override=None, gating_calibration_override=None, gating_diagnostics=None, exec_mode=None)` @ L3365 — Execute the rolling equity forecasting experiment.
  - `main()` @ L3814 — Entry point for CLI execution.

## `experiments/equity_panel/sweep_acceptance.py`
- module: `experiments.equity_panel.sweep_acceptance`
- classes:
  - `SweepParams` @ L42
- functions:
  - `_load_grid(arg)` @ L58
  - `_build_parameter_grid(grid_cfg)` @ L80
  - `_load_factor_returns(config)` @ L99
  - `_extract_metrics(run_dir, estimators)` @ L124
  - `run_sweep(args)` @ L197
  - `parse_args(argv=None)` @ L359
  - `main(argv=None)` @ L392

## `experiments/etf_panel/run.py`
- module: `experiments.etf_panel.run`
- classes:
  - `ETFConfig` @ L20
- functions:
  - `parse_args(argv=None)` @ L30
  - `main(argv=None)` @ L82

## `experiments/eval/config.py`
- module: `experiments.eval.config`
- classes:
  - `ResolveResult` @ L111
- functions:
  - `_deep_merge(base, override)` @ L15
  - `_load_yaml(path)` @ L25
  - `_load_json(path)` @ L35
  - `_normalise_layer(payload)` @ L45
  - `resolve_eval_config(args)` @ L116

## `experiments/eval/diagnostics.py`
- module: `experiments.eval.diagnostics`
- classes:
  - `DiagnosticReason` @ L6 (bases: str, Enum)
- functions: none

## `experiments/eval/inject_spike.py`
- module: `experiments.eval.inject_spike`
- classes:
  - `WindowSample` @ L28
- functions:
  - `_parse_float_list(raw, name)` @ L34
  - `_make_overlay_config(config)` @ L49
  - `_collect_windows(config, raw_returns, residuals, vol_proxy_full, *, factor_tracking_required, residual_index_set)` @ L75
  - `_baseline_fp(samples, overlay_cfg)` @ L160
  - `_inject_spike(matrix, rng, mu)` @ L169
  - `parse_args(argv=None)` @ L182
  - `main(argv=None)` @ L210

## `experiments/eval/run.py`
- module: `experiments.eval.run`
- classes:
  - `EvalOutputs` @ L85
  - `EvalConfig` @ L226
- functions:
  - `_plot_regime_histograms(diagnostics_df, column, *, out_dir, xlabel, title_prefix)` @ L99
  - `_format_group_label_counts(labels, design)` @ L291
  - `_vol_state_label(value, calm_cut, crisis_cut)` @ L332
  - `_serialise_config(config)` @ L342
  - `_current_git_sha()` @ L410
  - `_write_run_metadata(path, payload)` @ L422
  - `_aggregate_skip_stats(metrics_df, regime=None)` @ L427 — Aggregate per-method skip shares by reason for a given regime.
  - `_paths_to_strings(path_map)` @ L462
  - `_mode_string(values)` @ L466
  - `_aligned_error_table(metrics, regime, portfolio, *, column, estimator_ref='overlay', comparator='baseline', valid_window_ids=None)` @ L479 — Return per-window errors aligned on common availability.
  - `_aligned_dm_stat(metrics, regime, portfolio, *, column='sq_error', comparator='baseline', valid_window_ids=None, min_windows=MIN_COMPARISON_WINDOWS)` @ L523
  - `_aligned_delta_mean(metrics, regime, portfolio, *, column, comparator='baseline', valid_window_ids=None)` @ L552 — Aligned mean difference overlay - comparator for a given loss column.
  - `_apply_multi_alignment_guard(detections, *, threshold, max_keep)` @ L579 — Drop detections beyond the first if alignment cosine falls below threshold.
  - `_sign_test_stat(aligned, comparator)` @ L607
  - `_bootstrap_delta_mse(diffs, resamples, rng, block_size=None)` @ L631
  - `_vol_thresholds(vol_proxy, train_end, config)` @ L665
  - `parse_args(argv=None)` @ L680
  - `_compute_vol_proxy(returns, span=21)` @ L1130
  - `_write_overlay_toggle(path, summary)` @ L1138
  - `_plot_histogram(series, path, *, xlabel, title, bins=20)` @ L1170
  - `_plot_acceptance_edge_histograms(diagnostics_df, design, out_dir)` @ L1197
  - `_detail_defaults()` @ L1227
  - `_safe_nanmean(values)` @ L1246
  - `_safe_nanmedian(values)` @ L1256
  - `_top_mean(values, count)` @ L1266
  - `_safe_share(successes, total)` @ L1277
  - `_required_replicates(design, config)` @ L1283
  - `_build_grouped_window(frame, *, config, calm_threshold, crisis_threshold, vol_proxy)` @ L1294
  - `_min_variance_weights(covariance, *, ridge=None, box=(0.0, 1.0), cache=None, gamma=None, tau=0.0, prev_weights=None, solver='projgrad', solver_name=None, skip_on_missing_solver=False)` @ L1329
  - `_expected_shortfall(sigma, alpha=0.05)` @ L1403
  - `_realised_tail_mean(returns, var_threshold)` @ L1408
  - `_safe_condition_number(matrix)` @ L1415
  - `_qlike_loss(forecast_var, realised_var)` @ L1422
  - `_limit_windows_by_regime(metrics_df, diagnostics_df, *, calm_limit, crisis_limit, seed)` @ L1434
  - `_window_regime(vol_proxy, date, calm_cut, crisis_cut, *, fallback=None)` @ L1494
  - `_prepare_returns(config)` @ L1522
  - `run_evaluation(config, *, resolved_config=None, forced_changed_windows=None)` @ L1627
  - `main(argv=None)` @ L3773

## `experiments/eval/sensitivity.py`
- module: `experiments.eval.sensitivity`
- classes:
  - `Combo` @ L32
- functions:
  - `_parse_bool_grid(raw)` @ L51
  - `_parse_alignment_grid(raw)` @ L66
  - `_parse_float_grid(raw, name)` @ L84
  - `_ensure_matplotlib()` @ L99
  - `_thread_env()` @ L106
  - `_run_evaluation(command, env)` @ L128
  - `_build_command(base_args, combo, run_dir)` @ L132
  - `_load_first_row(path)` @ L186
  - `_changed_window_ids(detail_source)` @ L193
  - `_mean_delta_sq_error(metrics_path, changed_ids, portfolio)` @ L211
  - `_dm_stats_from_metrics(metrics_path, changed_ids, portfolio)` @ L233
  - `_plot_heatmap(subset, delta_values, stability_values, metric, title, xlabel, ylabel, path)` @ L265
  - `_full_regime(detail_df)` @ L305
  - `_metric_series(detail_df, column, changed_ids)` @ L315
  - `_median_metric(detail_df, column, changed_ids)` @ L334
  - `_plot_metric_histograms(metric_map, *, figures_dir, slug, xlabel, title_prefix)` @ L341
  - `parse_args(argv=None)` @ L379
  - `main(argv=None)` @ L413

## `experiments/prewhiten.py`
- module: `experiments.prewhiten`
- classes:
  - `PrewhitenTelemetry` @ L26
- functions:
  - `identity_prewhiten_result(returns, factor_cols=None)` @ L37
  - `select_prewhiten_factors(factors, requested)` @ L72
  - `_beta_abs_stats(betas)` @ L97
  - `compute_prewhiten_telemetry(whitening, *, requested_mode, effective_mode)` @ L113
  - `apply_prewhitening(returns, *, factors, requested_mode)` @ L140
  - `write_prewhiten_diagnostics(out_dir, whitening, telemetry)` @ L169

## `experiments/synthetic/__init__.py`
- module: `experiments.synthetic`
- classes: none
- functions: none

## `experiments/synthetic/calibrate_thresholds.py`
- module: `experiments.synthetic.calibrate_thresholds`
- classes: none
- functions:
  - `_parse_float_list(values)` @ L29
  - `parse_args(argv=None)` @ L41
  - `build_planned_jobs(p_assets_list, n_groups_list, replicates_list, delta_abs_grid)` @ L181 — Return the Cartesian product of sweep dimensions (excluding edge modes).
  - `_parse_bins(specs, *, prefix)` @ L205
  - `_assign_bin(value, bins, *, default_prefix)` @ L229
  - `_maybe_plot(entries, alpha, path)` @ L236
  - `_cell_identifier(p_assets, n_groups, replicates, delta_abs, edge_mode)` @ L284
  - `_load_shard_jobs(manifest_path, shard_id)` @ L288
  - `_build_cell_records(config, result, edge_mode)` @ L318
  - `_write_cell_payload(path, payload)` @ L355
  - `_load_cell_payloads(cells_dir)` @ L362
  - `_collect_cell_records(cell_payloads)` @ L374
  - `_build_threshold_map(entries, replicate_bins, asset_bins, alpha)` @ L385
  - `_build_defaults_payload(thresholds_map, *, alpha, thresholds_path)` @ L454
  - `main(argv=None)` @ L519
  - `_git_sha()` @ L766
  - `_blas_info()` @ L775
  - `_instance_metadata()` @ L788

## `experiments/synthetic/harness_utils.py`
- module: `experiments.synthetic.harness_utils`
- classes:
  - `HarnessConfig` @ L27 — Configuration for synthetic null/power harness simulations.
  - `ScoreResult` @ L53 — Container for per-trial spectral scores.
  - `SimulatedScores` @ L79 — Structured return for score simulations.
  - `EnergyFloorSelection` @ L224
- functions:
  - `_compute_scatter(observations, edge_mode)` @ L91
  - `_score_trial(observations, edge_mode)` @ L106
  - `_run_single_mu(config, mu, *, scenario_label)` @ L121
  - `simulate_scores(config, mu_values, *, scenario_prefix='')` @ L163 — Simulate score distributions for the supplied spike strengths.
  - `roc_table(null_scores, power_scores, *, thresholds=None)` @ L181 — Return a ROC-style table (FPR vs power) per edge mode and spike.
  - `select_energy_floor(null_scores, power_scores, *, target_fpr)` @ L241 — Select an energy floor that satisfies the FPR cap while maximising power.
  - `write_run_metadata(path, *, config, extra=None)` @ L300

## `experiments/synthetic/nested_killtest.py`
- module: `experiments.synthetic.nested_killtest`
- classes:
  - `TrialResult` @ L73
- functions:
  - `_wilson_interval(successes, trials, alpha=0.05)` @ L86 — Wilson score interval for a Bernoulli proportion.
  - `_current_git_sha()` @ L101
  - `_nan_safe(val, default)` @ L113
  - `load_config(path)` @ L117
  - `simulate_nested_panel(rng, *, n_assets, years, weeks, replicates, spike_strength, signal_to_noise, noise_variance)` @ L128 — Return (observations, year_labels, week_labels).
  - `_edge_scale(observations, edge_mode, edge_huber_c)` @ L173
  - `_gate_nested_detections(detections, config, *, delta_frac_used)` @ L212 — Apply overlay-like gating to nested detections.
  - `run_trials(config)` @ L273
  - `summarise_results(results)` @ L462
  - `write_summary_markdown(summary_df, out_path)` @ L494
  - `_candidate_metrics(summary_df, target_fpr)` @ L509
  - `_select_best_candidate(candidates, target_fpr)` @ L541
  - `_write_calibration_file(path, *, selection, sweep, summary_df, config)` @ L564
  - `main(argv=None)` @ L634

## `experiments/synthetic/null.py`
- module: `experiments.synthetic.null`
- classes: none
- functions:
  - `parse_args(argv=None)` @ L32
  - `_git_sha()` @ L63
  - `_build_fpr_curve(scores)` @ L70
  - `_plot_fpr_curve(curve, path)` @ L89
  - `main(argv=None)` @ L104

## `experiments/synthetic/power.py`
- module: `experiments.synthetic.power`
- classes: none
- functions:
  - `parse_args(argv=None)` @ L40
  - `_git_sha()` @ L95
  - `_load_null_scores(path)` @ L102
  - `_plot_roc(roc, mu_values, selection, path)` @ L115
  - `_save_defaults(path, *, selection, config, args, mu_values)` @ L145
  - `main(argv=None)` @ L180

## `experiments/synthetic/power_null.py`
- module: `experiments.synthetic.power_null`
- classes:
  - `TrialResult` @ L200
- functions:
  - `_resolve_delta_grid(delta_grid)` @ L76
  - `_normalise_for_meta(value)` @ L83
  - `_normalise_panel_specs(panel_specs)` @ L95
  - `calibration_cache_meta(*, config, edge_modes, alpha, trials_null, delta_grid, panel_specs=None)` @ L107
  - `load_calibration_cache(path, meta, dependencies=None)` @ L130
  - `write_calibration_cache(path, payload, meta)` @ L161
  - `_edge_scale_for_mode(y, mode, huber_c=1.5)` @ L169
  - `_detections_for_mode(y, groups, *, edge_mode, gating, config, delta_frac_override=None)` @ L210
  - `_simulate_null(config, *, rng)` @ L255
  - `_simulate_power(config, *, rng, strength, two_spike)` @ L267
  - `run_trials(*, config, edge_modes, gating_labels, trials_null, trials_power, spike_grid, two_spike, rng)` @ L297
  - `calibrate_delta_thresholds(*, config, edge_modes, trials_null, alpha, rng, delta_grid=None, panel_specs=None)` @ L370 — Estimate minimal delta_frac values achieving target null FPR for each (p, T).
  - `summarise_results(results)` @ L481
  - `plot_fpr_heatmap(summary, out_path)` @ L507
  - `plot_power_curves(summary, out_path)` @ L527
  - `parse_args()` @ L550
  - `main()` @ L644

## `experiments/synthetic_oneway/__init__.py`
- module: `experiments.synthetic_oneway`
- classes: none
- functions: none

## `experiments/synthetic_oneway/run.py`
- module: `experiments.synthetic_oneway.run`
- classes: none
- functions:
  - `load_config(path)` @ L69 — Load experiment configuration from YAML, falling back to defaults.
  - `ensure_dir(path)` @ L83 — Create ``path`` and parents if they do not yet exist.
  - `simulate_panel(rng, *, n_assets, n_groups, replicates, spike_strength, noise_variance, signal_to_noise, return_dirs=False)` @ L89 — Simulate a balanced MANOVA panel with a single spike.
  - `simulate_multi_spike(rng, *, n_assets, n_groups, replicates, spike_strengths, noise_variance, signal_to_noise)` @ L130 — Simulate a panel with multiple planted spikes.
  - `mp_upper_edge(noise_variance, n_assets, n_groups)` @ L170 — Return the Marčenko–Pastur upper edge for the supplied regime.
  - `histogram_s1(eigenvalues, edge, out_dir)` @ L177 — Save the S1 histogram visualising the empirical spectrum.
  - `bias_table_s3(df, out_dir)` @ L197 — Persist the S3 bias summary to disk.
  - `s2_vector_alignment(config, rng)` @ L203 — Evaluate alignment between the leading eigvector and the planted spike.
  - `summary_to_json(summary, out_dir)` @ L252 — Write a JSON summary of the synthetic experiments.
  - `s1_monte_carlo(config, rng)` @ L259 — Run the S1 Monte Carlo sweep and return summary statistics.
  - `s3_bias(config, rng)` @ L295 — Evaluate aliased versus de-aliased bias across spike strengths.
  - `s4_guardrail_analysis(config, rng)` @ L373 — Compare false-positive rates under default versus lax guardrails.
  - `s5_multi_spike_bias(config, rng)` @ L489 — Assess bias reduction in a multi-spike setting.
  - `plot_bias_timeseries(prefixes, aliased, dealiased, spike, output_dir)` @ L695 — Plot a diagnostic bias timeseries for an individual spike size.
  - `run_experiment(config_path=None, *, seed=None, progress=None)` @ L716 — Execute the S1/S3 synthetic experiments.
  - `main()` @ L764 — Entry point for CLI execution.

## `src/baselines/__init__.py`
- module: `baselines`
- classes: none
- functions: none

## `src/baselines/covariance.py`
- module: `baselines.covariance`
- classes: none
- functions:
  - `_symmetrize(matrix)` @ L21
  - `sample_covariance(observations)` @ L25 — Sample covariance with Bessel correction.
  - `lw_covariance(observations)` @ L38 — Ledoit–Wolf shrinkage covariance estimator.
  - `oas_covariance(observations)` @ L44 — Oracle Approximating Shrinkage (OAS) estimator.
  - `cc_covariance(observations)` @ L50 — Ledoit–Wolf constant-correlation shrinkage estimator.
  - `rie_covariance(sample_covariance, *, sample_count=None)` @ L56 — Rotationally-invariant estimator (RIE) shrinkage towards the spectrum mean.
  - `quest_covariance(sample_covariance, *, sample_count)` @ L81 — QuEST-style spectral clipping based on Marchenko–Pastur support.
  - `ewma_covariance(observations, *, halflife=30.0, centre=True)` @ L112 — Exponentially weighted moving-average covariance estimate.

## `src/baselines/factors.py`
- module: `baselines.factors`
- classes:
  - `PrewhitenResult` @ L44
- functions:
  - `_normalise_columns(frame)` @ L53
  - `_detect_percentage_scale(frame)` @ L61
  - `_load_candidate(path)` @ L74
  - `load_observed_factors(*, returns=None, path=None, data_dir=None, required=None)` @ L105 — Load observed factor returns, preferring FF5+MOM datasets when available.
  - `_prepare_design_matrix(index, factors, *, add_intercept)` @ L158
  - `_align_returns_factors(returns, factors, *, dropna)` @ L174
  - `prewhiten_returns(returns, factors, *, add_intercept=True, dropna=True)` @ L206 — Regress asset returns on observed factors and return residual series.

## `src/eval/__init__.py`
- module: `eval`
- classes: none
- functions: none

## `src/eval/balance.py`
- module: `eval.balance`
- classes:
  - `BalanceTelemetry` @ L13
  - `BalanceResult` @ L25
- functions:
  - `build_balanced_window(frame, group_labels, *, min_replicates)` @ L32

## `src/eval/clean.py`
- module: `eval.clean`
- classes:
  - `NaNPolicyTelemetry` @ L13
  - `NaNPolicyResult` @ L23
- functions:
  - `apply_nan_policy(frame, group_labels, *, max_missing_asset, max_missing_group_row)` @ L29

## `src/evaluation/__init__.py`
- module: `evaluation`
- classes: none
- functions:
  - `check_dealiased_applied(estimates)` @ L21 — Assert de-aliased forecasts differ from aliased when detections exist.

## `src/evaluation/dm.py`
- module: `evaluation.dm`
- classes: none
- functions:
  - `_newey_west_long_run_variance(diffs, lags)` @ L10
  - `dm_test(err1, err2, *, h=1, use_nw=True, lags=None)` @ L27 — Diebold–Mariano test for equal predictive accuracy.

## `src/evaluation/evaluate.py`
- module: `evaluation.evaluate`
- classes:
  - `DeltaSummary` @ L367
- functions:
  - `iqr(values)` @ L32 — Interquartile range (75th - 25th percentile).
  - `sign_test_pvalue(differences)` @ L53 — Two-sided sign test p-value for paired differences.
  - `qlike(forecasts, realised)` @ L86 — Quasi-likelihood (QLIKE) loss for variance forecasts.
  - `block_bootstrap_ci_median(series, *, block_len=12, n_boot=1000, alpha=0.05, rng=None)` @ L102 — Moving block bootstrap CI for the median of a time series.
  - `_clip_prob(value, eps=1e-08)` @ L163
  - `kupiec_pof_test(violations, alpha=0.05)` @ L167 — Return the Kupiec proportion-of-failures p-value.
  - `christoffersen_independence_test(violations)` @ L184 — Return the Christoffersen independence test p-value.
  - `expected_shortfall_test(losses, es_forecasts, violations)` @ L214 — Approximate two-sided t-test comparing realised losses and ES forecasts.
  - `alignment_diagnostics(covariance, direction, *, top_p=3)` @ L239 — Return (angle_deg, energy_mu) between detection direction and PCA subspace.
  - `plot_variance_error_panel(errors, base_path)` @ L274 — Plot E3: variance MSE mean and distribution by method.
  - `plot_coverage_error(coverage_errors, base_path)` @ L330 — Plot E4: VaR(95%) coverage errors by method.
  - `summarize_deltas(deltas, *, block_len=12, n_boot=1000, alpha=0.05, rng=None)` @ L375 — Return robust summary statistics and CI for paired deltas.
  - `build_metrics_summary(*, errors_by_combo, coverage_errors, qlike_by_combo=None, var_forecasts=None, realised_returns=None, es_forecasts=None, label, block_len=12, n_boot=1000, alpha=0.05)` @ L402 — Aggregate window-level errors into a metrics summary DataFrame.

## `src/evaluation/factor.py`
- module: `evaluation.factor`
- classes:
  - `POETResult` @ L42
- functions:
  - `observed_factor_covariance(returns, factors, *, add_intercept=True)` @ L13 — Estimate Σ = B Σ_f Bᵀ + Σ_ε from observed factor returns via cross-sectional OLS.
  - `_poet_ic(residual_var, k, p, n)` @ L47
  - `poet_lite_covariance(returns, *, max_factors=10, shrink='diag')` @ L53 — Estimate a POET-lite covariance using PCA loadings with simple residual shrinkage.

## `src/finance/__init__.py`
- module: `finance`
- classes: none
- functions: none

## `src/finance/design.py`
- module: `finance.design`
- classes: none
- functions:
  - `build_design_matrix(returns, factors)` @ L7 — Join returns with factor realisations on their common timeline.
  - `groups_from_weeks(index)` @ L34 — Assign an integer group id to each timestamp based on its week.

## `src/finance/eval.py`
- module: `finance.eval`
- classes: none
- functions:
  - `rolling_windows(panel, window_weeks, horizon_weeks)` @ L21 — Yield expanding fit/hold windows over the weekly panel.
  - `risk_metrics(forecasts, realised)` @ L52 — Compute mean squared error and 95% VaR coverage error.
  - `oos_variance_forecast(y_fit, y_hold, w, estimator, **kwargs)` @ L84 — Compute out-of-sample variance forecasts and realised variance.
  - `weekly_cov_from_components(ms1, ms2, replicates, mu_hats=None, vecs=None, clip_top=None)` @ L223 — Construct the weekly covariance of summed daily returns from MANOVA components.
  - `variance_forecast_from_components(y_fit, y_hold, replicates, w, detections=None)` @ L298 — Forecast portfolio variance from balanced MANOVA components and compare to realised.
  - `evaluate_portfolio(returns, weights)` @ L400 — Compute realised return and volatility for the supplied weights.

## `src/finance/factors.py`
- module: `finance.factors`
- classes: none
- functions:
  - `_align_frames(returns, factors, industry)` @ L10 — Align inputs on their shared date index and drop factor-side NaNs.
  - `_prepare_design(factors, industry)` @ L43 — Combine factor and industry data into a single numeric design matrix.
  - `factor_covariance(R_df, F_df, *, add_intercept=True, industry_df=None)` @ L64 — Estimate an observed-factor covariance matrix via cross-sectional OLS.

## `src/finance/io.py`
- module: `finance.io`
- classes: none
- functions:
  - `load_prices_csv(path)` @ L15 — Load a tidy price history CSV with canonical dtypes.
  - `to_daily_returns(price_frame)` @ L52 — Convert tidy prices to a wide matrix of daily log returns.
  - `load_market_data(path, *, parse_dates=True)` @ L79 — Backward-compatible alias for :func:`load_prices_csv`.
  - `load_returns_csv(path)` @ L99 — Load a tidy daily returns CSV into a wide date-indexed matrix.

## `src/finance/ledoit.py`
- module: `finance.ledoit`
- classes: none
- functions:
  - `lw_cov(x)` @ L10 — Compute the Ledoit–Wolf covariance estimate.
  - `ledoit_wolf_shrinkage(x)` @ L35 — Backward-compatible alias for :func:`lw_cov`.

## `src/finance/loader.py`
- module: `finance.loader`
- classes:
  - `WeeklyLoadResult` @ L14
- functions:
  - `_balanced_weekly_from_daily(daily_returns, replicates=5)` @ L20 — Internal helper: balanced weekly panel with a fixed universe.
  - `load_weekly_from_daily_csv(path, *, start=None, end=None, min_p=50)` @ L77 — Load daily prices CSV, build balanced weekly panel, and print counters.
  - `rolling_windows_fixed_universe(weekly, *, window_weeks, horizon_weeks, min_p=50)` @ L108 — Yield (fit, hold) windows with per-window fixed-universe enforcement.

## `src/finance/portfolio.py`
- module: `finance.portfolio`
- classes:
  - `MinVarMemo` @ L58 — Cache penalised covariance factorizations per window.
- functions:
  - `_symmetrize(matrix)` @ L11
  - `_project_box_sum(v, lo, hi, target)` @ L15
  - `minvar_ridge_box(Sigma, *, box=(0.0, 1.0), ridge=0.0001, sum_to_one=True, max_iter=3000, tol=1e-07, cache=None)` @ L88 — Projected-gradient minimum-variance solver with ridge and box bounds.
  - `turnover(w_prev, w_new)` @ L164 — Compute one-way turnover between consecutive portfolios.
  - `apply_turnover_cost(var_series, w_series, bps)` @ L174 — Apply turnover costs (in basis points) to a variance or PnL series.

## `src/finance/portfolios.py`
- module: `finance.portfolios`
- classes:
  - `MissingSolverError` @ L10 (bases: RuntimeError) — Raised when a required optimisation solver dependency is unavailable.
  - `OptimizationResult` @ L31 — Result container for portfolio optimisation routines.
- functions:
  - `_get_cvxpy()` @ L14
  - `equal_weight(p)` @ L43 — Return the equal-weight vector for ``p`` assets.
  - `_solve_min_variance_cvxpy(covariance, *, allow_short=False, box=None, ridge=0.0, solver=None)` @ L62 — Solve the minimum-variance problem using cvxpy.
  - `minimum_variance(covariance, *, allow_short=False, solver=None, ridge=0.0)` @ L130 — Solve the minimum-variance problem using cvxpy (if available).
  - `min_variance_box(covariance, lb=-0.02, ub=0.02, *, solver=None, ridge=0.0)` @ L148 — Solve the minimum-variance problem with box constraints.
  - `optimize_portfolio(covariance, target_return=None, *, allow_short=False, skip_on_missing_solver=False, box=None, ridge=0.0, solver=None)` @ L169 — Return the minimum-variance portfolio; fail loud if solver is missing by default.

## `src/finance/returns.py`
- module: `finance.returns`
- classes: none
- functions:
  - `compute_log_returns(prices)` @ L11 — Compute log returns for a wide price DataFrame.
  - `weekly_panel(daily_returns, start, end)` @ L35 — Aggregate daily log returns into weekly (Monday-start) log returns.
  - `balance_weeks(panel)` @ L80 — Create a balanced week/day design from daily returns.

## `src/finance/robust.py`
- module: `finance.robust`
- classes: none
- functions:
  - `winsorize(returns_df, q)` @ L7 — Clip each column of ``returns_df`` to its [q, 1-q] empirical quantiles.
  - `huberize(returns_df, c)` @ L20 — Apply column-wise Huber clipping using median and MAD scale.
  - `tyler_shrink_covariance(observations, *, ridge=0.001, max_iter=200, tol=1e-06)` @ L41 — Return a Tyler M-estimator with ridge regularisation for positive definiteness.

## `src/finance/shrinkage.py`
- module: `finance.shrinkage`
- classes: none
- functions:
  - `_validate_input(R)` @ L13
  - `_sample_covariance(X)` @ L22
  - `_symmetrize(matrix)` @ L28
  - `_warn_and_fill_nonfinite(name, data)` @ L32
  - `_assert_psd_and_symmetric(name, matrix)` @ L50
  - `oas_covariance(R)` @ L58 — Oracle Approximating Shrinkage covariance targeting the identity matrix.
  - `cc_covariance(R)` @ L70 — Ledoit–Wolf constant-correlation shrinkage covariance estimator.

## `src/fjs/__init__.py`
- module: `fjs`
- classes: none
- functions:
  - `_missing_matplotlib(*_args, **_kwargs)` @ L12

## `src/fjs/balanced.py`
- module: `fjs.balanced`
- classes:
  - `BalancedConfig` @ L11 — Configuration for the balanced risk contribution solver.
- functions:
  - `_validate_balanced_inputs(y, groups)` @ L30 — Return validated observations and grouping assignments for a balanced design.
  - `_compute_group_means(observations, inverse, counts)` @ L56 — Accumulate per-group means using the grouping inverse index.
  - `group_means(y, groups)` @ L69 — Compute per-group and overall means for a balanced one-way MANOVA design.
  - `mean_squares(y, groups)` @ L95 — Estimate balanced one-way MANOVA mean squares and covariance components.
  - `compute_balanced_weights(returns, config)` @ L152 — Compute portfolio weights that balance the contribution of estimated MANOVA spikes.

## `src/fjs/balanced_nested.py`
- module: `fjs.balanced_nested`
- classes:
  - `NestedDesignMetadata` @ L11 — Balanced nested Year⊃Week design metadata.
- functions:
  - `_validate_labels(labels, name, expected_length)` @ L24
  - `mean_squares_nested(y, year_labels, week_of_year_labels, replicates)` @ L37 — Compute balanced nested Year⊃Week MANOVA mean squares.

## `src/fjs/dealias.py`
- module: `fjs.dealias`
- classes:
  - `DesignParams` @ L16 (bases: TypedDict)
  - `Detection` @ L24 (bases: TypedDict)
  - `DealiasingResult` @ L61 — Container for the results of spectral de-aliasing.
- functions:
  - `_compute_admissible_root(lam_val, a_vec, C_for_mp, d_vec, n_total, cs_vec)` @ L69
  - `_orthonormal_tangent_basis(a_vec)` @ L91 — Return an orthonormal basis for the tangent space at ``a_vec`` on the sphere.
  - `_rotate_on_sphere(base, tangent, angle)` @ L119
  - `_generate_unit_vectors(component_count, a_grid, *, nonnegative)` @ L131
  - `_normalise_angle(theta)` @ L182
  - `_angle_key(theta)` @ L190
  - `_sigma_of_a_from_MS(a, MS_list)` @ L195 — Return Σ̂(a)=∑_s a_s MS_s (balanced design).
  - `dealias_covariance(covariance, spectrum)` @ L216 — Remove aliasing artefacts from a sample covariance matrix.
  - `_validate_inputs(y, groups)` @ L310
  - `_default_design(stats)` @ L324
  - `_merge_detections(detections, eps_factor=0.05)` @ L343
  - `dealias_search(y, groups, target_r, *, Cs=None, a_grid=120, delta=0.5, delta_frac=None, eps=0.02, energy_min_abs=None, stability_eta_deg=1.0, use_tvector=True, nonnegative_a=False, design=None, cs_drop_top_frac=None, cs_sensitivity_frac=None, use_design_c_for_C=False, scan_basis='ms', oneway_a_solver='grid', off_component_leak_cap=0.3, cs_scale=None, diagnostics=None, stats=None, edge_scale=None, edge_mode=None)` @ L394 — Perform Algorithm 1 de-aliasing search for one-way balanced designs.

## `src/fjs/gating.py`
- module: `fjs.gating`
- classes: none
- functions:
  - `_as_float(value, *, default=float('nan'))` @ L15 — Best-effort conversion to float with NaN fallback.
  - `_score_detection(det)` @ L31 — Return score tuple (primary score, edge margin, lambda) for ordering.
  - `count_isolated_outliers(eigs, edge, stability=None)` @ L51 — Count isolated spikes relative to the MP edge and stability.
  - `select_top_k(detections, k)` @ L109 — Select the top-k detections ranked by score = energy * stability.
  - `_load_delta_thresholds(path_str)` @ L136 — Load the calibrated delta thresholds JSON with basic validation.
  - `lookup_calibrated_delta(edge_mode, p, t, *, calibration_path, design=None)` @ L149 — Return the calibrated delta_frac for the given (edge_mode, p, t) combo.

## `src/fjs/mp.py`
- module: `fjs.mp`
- classes:
  - `MarchenkoPasturModel` @ L94 — Summary statistics for a Marchenko–Pastur limiting law.
- functions:
  - `configure_mp_cache(directory)` @ L27 — Configure the on-disk MP edge cache directory at runtime.
  - `clear_mp_cache()` @ L40 — Clear the in-memory MP edge cache.
  - `_cache_get(key)` @ L46
  - `_cache_set(key, value)` @ L67
  - `_hash_arrays(*arrays)` @ L81
  - `_prepare_inputs(a, C, d, N)` @ L112
  - `_prepare_cs(Cs, template)` @ L133
  - `estimate_Cs_from_MS(MS_list, d_list, c_list, drop_top=1)` @ L147 — Estimate trace-based noise plug-ins C_s from the supplied mean squares.
  - `_k_values(a, C, d, N)` @ L211
  - `z_of_m(m, a, C, d, N, Cs=None)` @ L220 — Evaluate the closed-form Marčenko–Pastur z(m) transform.
  - `z0(m, a, C, d, N, Cs=None)` @ L254 — Balanced one-way z0(m) in closed form.
  - `_dz_dm(m, k_vals, numerators)` @ L277
  - `_d2z_dm2(m, k_vals, numerators)` @ L290
  - `z0_prime(m, a, C, d, N, Cs=None)` @ L303 — Closed-form first derivative z0'(m) for balanced one-way design.
  - `z0_double_prime(m, a, C, d, N, Cs=None)` @ L325 — Closed-form second derivative z0''(m) for balanced one-way design.
  - `_logspace_grid()` @ L346
  - `_augment_with_singularities(grid, k_vals)` @ L351
  - `_newton_refine(x0, f, fp, *, max_iter=8, tol=1e-14)` @ L367 — One-step Newton refinement with simple safeguards.
  - `_stationary_points(k_vals, numerators)` @ L398 — Locate stationary points of z(m) by bracketing zeros of z'(m).
  - `_bisect(func, left, right, *, max_iter=200, tol=1e-12)` @ L428
  - `_crosses_pole(m1, m2, k_vals)` @ L479 — Return True if the interval [m1, m2] crosses a pole 1 + k m = 0.
  - `_root_brackets(func, points, k_vals=None)` @ L491
  - `_brackets_sign_change(func, points, k_vals)` @ L525 — Find sign-change brackets while guarding against poles.
  - `mp_edge(a, C, d, N, Cs=None)` @ L558 — Locate the upper bulk edge of the Marčenko--Pastur distribution.
  - `_mp_edge_impl(a_arr, c_arr, d_arr, n_float, cs_arr)` @ L579
  - `m_edge(a, C, d, N, Cs=None)` @ L617 — Return m_plus where z'(m_plus)=0 and z''(m_plus)<0 (upper edge).
  - `admissible_m_from_lambda(lam, a, C, d, N, Cs=None)` @ L644 — Recover the admissible real root of z(m) = λ with positive slope.
  - `_admissible_m_from_lambda_impl(lam_val, a_arr, c_arr, d_arr, n_float, cs_arr)` @ L669
  - `_normalise_order(order, n_strata)` @ L735
  - `t_vec(lam, a, C, d, N, c, order, Cs=None)` @ L760 — Evaluate the t-vector associated with λ using the admissible root m(λ).
  - `marchenko_pastur_edges(model)` @ L800 — Compute the theoretical support edges for a Marchenko–Pastur distribution.
  - `marchenko_pastur_pdf(model, grid)` @ L817 — Evaluate the Marchenko–Pastur density over a grid.
  - `scale_Cs(Cs, alpha)` @ L839 — Return a scaled copy of the Cs plug-ins by factor ``alpha``.

## `src/fjs/overlay.py`
- module: `fjs.overlay`
- classes:
  - `OverlayConfig` @ L24
- functions:
  - `_bracket_status_label(detections)` @ L52
  - `_summarise_pre_gate(detections, cfg)` @ L73
  - `_coarse_candidates(observations, cfg)` @ L108
  - `_baseline_covariance(sample_covariance, *, observations, config)` @ L193
  - `_resolve_delta_frac(cfg, observations, groups)` @ L236
  - `_gate_detections(detections, cfg, soft_cap, delta_frac_used)` @ L256
  - `detect_spikes(observations, groups, *, config=None, stats=None)` @ L324
  - `apply_overlay(sample_covariance, detections, *, observations=None, config=None, baseline_covariance=None)` @ L400

## `src/fjs/robust.py`
- module: `fjs.robust`
- classes: none
- functions:
  - `_ensure_2d(x)` @ L20
  - `_symmetrize(matrix)` @ L28
  - `_initial_scatter(x)` @ L32
  - `tyler_scatter(observations, *, max_iter=200, tol=1e-06, ridge=1e-06)` @ L47 — Return the Tyler fixed-point scatter estimate with optional ridge regularisation.
  - `huber_scatter(observations, c, *, max_iter=100, tol=1e-06, ridge=1e-06)` @ L109 — Compute a Huber-type reweighted scatter estimator.
  - `edge_from_scatter(scatter, n_features, n_samples)` @ L178 — Estimate the upper Marčenko–Pastur edge from a scatter matrix.

## `src/fjs/spectra.py`
- module: `fjs.spectra`
- classes: none
- functions:
  - `topk_eigh(matrix, k)` @ L16 — Return the largest ``k`` eigenpairs of a symmetric matrix.
  - `project_alignment(vector, subspace)` @ L48 — Compute the projection norm of ``vector`` onto the span of ``subspace``.
  - `_ensure_path(path)` @ L80
  - `plot_spectrum_with_edges(eigenvalues, edges, out_path, *, title=None, xlabel='Eigenvalue index (descending)', ylabel='Eigenvalue', highlight_threshold=None, highlight_color='C3')` @ L86 — Plot an empirical spectrum together with optional reference edge lines.
  - `plot_spike_timeseries(time_index, aliased_series, dealiased_series, out_path, *, title=None, true_value=None, xlabel='Prefix groups', ylabel='Spike magnitude')` @ L161 — Plot aliased and de-aliased spike estimates against a time index.
  - `estimate_spectrum(eigenvalues, *, bandwidth=None)` @ L219 — Return a sorted copy of ``eigenvalues`` (placeholder estimator).

## `src/fjs/theta_solver.py`
- module: `fjs.theta_solver`
- classes:
  - `ThetaSolverParams` @ L12 — Closed-form parameters required for the θ root-finding routine.
- functions:
  - `_normalise_angle(theta)` @ L28 — Return θ reduced to the principal interval [0, 2π).
  - `solve_theta_for_t2_zero(lambda_hat, params)` @ L34 — Solve for θ such that t₂(λ̂, θ) = 0 for k=2 balanced designs.

## `src/io/crsp_daily.py`
- module: `io.crsp_daily`
- classes:
  - `CrspQueryParams` @ L23 — Configuration for the CRSP daily snapshot.
- functions:
  - `explain_rowcount(sql, *, connection)` @ L76 — Return the planner's estimated rowcount for the provided query.
  - `_clean_snapshot(frame)` @ L97
  - `fetch_crsp_daily_snapshot(out_path, *, params=None)` @ L137 — Fetch CRSP daily snapshot and persist to parquet.
  - `build_dow_vol_labels(returns, *, ewma_span=21, calm_quantile=0.2, crisis_quantile=0.8)` @ L154 — Compute day-of-week and volatility-state labels.
  - `write_labels_parquet(labels, out_path)` @ L202

## `src/io/wrds_connect.py`
- module: `io.wrds_connect`
- classes: none
- functions:
  - `wrds_conn()` @ L8

## `src/meta/cache.py`
- module: `meta.cache`
- classes: none
- functions:
  - `window_key(manifest, week_list, tickers, replicates, *, code_signature=None, design=None, nested_replicates=None, oneway_a_solver=None, estimator=None, preprocess_flags=None)` @ L13 — Stable hash identifying a per-window cache entry.
  - `save_window(cache_dir, key, payload)` @ L52 — Persist cached per-window statistics.
  - `load_window(cache_dir, key)` @ L77 — Load cached per-window statistics if available.

## `src/meta/completeness.py`
- module: `meta.completeness`
- classes:
  - `CompletenessResult` @ L38
- functions:
  - `_load_json(path)` @ L9 — Read JSON from ``path`` if it exists; otherwise return an empty mapping.
  - `_coerce_int(value)` @ L21
  - `_first(mapping, keys)` @ L30
  - `_window_stats_from_manifest(manifest)` @ L72
  - `evaluate_eval_run(run_dir, *, label=None, require_manifest=True, allow_unknown_coverage=False, run_type='daily')` @ L96 — Assess completeness for a daily overlay evaluation (rc-lite/rc) run directory.
  - `_locate_payload_dir(base)` @ L154 — Return the most likely payload directory (handles tagged weekly outputs).
  - `evaluate_weekly_run(run_dir, *, label=None)` @ L167 — Assess completeness for a weekly equity_panel run directory.

## `src/meta/run_meta.py`
- module: `meta.run_meta`
- classes:
  - `RunMeta` @ L14 — Lightweight metadata summary for a single run.
- functions:
  - `code_signature(targets=None)` @ L74 — Compute a SHA-256 signature over core de-aliasing code.
  - `_git_sha()` @ L112 — Return the short git SHA for the current repository, or 'unknown'.
  - `_sha256_of_file(path)` @ L122
  - `_collect_pdf_hashes(directory)` @ L130
  - `_load_optional_json(path)` @ L142
  - `_count_detections(det_summary_path)` @ L152 — Return (detections_total, L_max) from detection_summary.csv if present.
  - `write_run_meta(output_dir, *, config=None, delta=None, delta_frac=None, a_grid=None, signed_a=None, sigma2_plugin=None, code_signature_hash=None, estimator=None, design=None, nested_replicates=None, preprocess_flags=None, label=None, crisis_label=None, solver_used=None, edge_mode=None, exec_mode=None)` @ L168 — Create a run_meta.json artifact in ``output_dir``.

## `src/meta/runtime.py`
- module: `meta.runtime`
- classes:
  - `ExecModeSettings` @ L26 — Resolved execution-mode settings shared across runners.
- functions:
  - `_set_threadpool_limits(max_threads)` @ L34
  - `_apply_thread_caps(max_threads)` @ L49
  - `resolve_exec_mode(mode, *, throughput_threads=None)` @ L56 — Return the execution-mode settings without mutating global state.
  - `configure_exec_mode(mode, *, throughput_threads=None)` @ L72 — Resolve and apply execution-mode thread caps.
  - `effective_worker_count(settings, requested_workers, cpu_count=None)` @ L80 — Return the worker count respecting the resolved execution mode.
  - `thread_caps_snapshot()` @ L98 — Return the current BLAS/OpenMP thread caps for logging.
  - `exec_mode_metadata(settings)` @ L104 — Helper to expose execution-mode metadata for run.json payloads.

## `src/plotting/__init__.py`
- module: `plotting`
- classes: none
- functions: none

## `src/plotting/utils.py`
- module: `plotting.utils`
- classes: none
- functions:
  - `_figures_dir_for_run(run)` @ L19 — Return figures directory for a given run under experiments/<run>/figures.
  - `e1_plot_spectrum_with_mp(eigenvalues, mp_edges, *, run, title=None)` @ L27 — E1: Plot spectrum with MP edge and mark outliers.
  - `e2_plot_spike_timeseries(time_index, aliased_series, dealiased_series, *, run, title=None, xlabel='Window', ylabel='Spike magnitude')` @ L62 — E2: Plot aliased vs de-aliased spike time-series.
  - `e3_plot_var_mse(errors_by_method, *, run, title='E3: Variance forecast MSE (mean)', ylabel='Squared error')` @ L101 — E3: Single-chart Var-MSE comparison across methods (bar of means).
  - `e4_plot_var_coverage(coverage_errors, *, run)` @ L150 — E4: VaR(95%) coverage error plot.
  - `s4_plot_guardrails_from_csv(csv_path, *, run, title='S4: Guardrails on isotropic data')` @ L167 — S4: Plot guardrail false-positive comparison from a CSV.

## `src/report/__init__.py`
- module: `report`
- classes: none
- functions: none

## `src/report/gather.py`
- module: `report.gather`
- classes: none
- functions:
  - `load_run(path)` @ L28 — Load core artifacts for a single run directory.
  - `find_runs(root, pattern=None)` @ L72 — Discover run directories, preferring tagged folders.
  - `_extract_detection(summary_df)` @ L98
  - `_extract_edge_stats(summary_df)` @ L104
  - `_dm_values(de_row, estimator)` @ L119
  - `_dm_values_qlike(de_row, estimator)` @ L130
  - `_ci_bounds(de_row, estimator)` @ L141
  - `collect_estimator_panel(run_paths)` @ L164 — Combine estimator diagnostics across runs into a single table.

## `src/report/plots.py`
- module: `report.plots`
- classes: none
- functions:
  - `_single_run_tag(df)` @ L23
  - `_ensure_dir(path)` @ L29
  - `plot_dm_bars(df, *, root=DEFAULT_FIG_ROOT)` @ L33
  - `plot_edge_margin_hist(df, *, root=DEFAULT_FIG_ROOT)` @ L73
  - `plot_detection_rate(df, *, root=DEFAULT_FIG_ROOT)` @ L102
  - `plot_alignment_angles(df, *, root=DEFAULT_FIG_ROOT)` @ L127
  - `plot_ablation_heatmap(df, *, root=DEFAULT_FIG_ROOT)` @ L157

## `src/report/tables.py`
- module: `report.tables`
- classes: none
- functions:
  - `_single_run_tag(df)` @ L19
  - `_ensure_dir(path)` @ L25
  - `_format_float(value)` @ L29
  - `_write_markdown(df, path)` @ L37
  - `_find_strategy_row(group, candidates)` @ L51
  - `table_estimators_panel(df, *, root=DEFAULT_FIG_ROOT)` @ L62 — Create estimator panel comparison tables and return paths to CSV, Markdown, and LaTeX outputs.
  - `table_rejections(df, *, root=DEFAULT_FIG_ROOT)` @ L191 — Generate a rejection reason summary table.
  - `table_ablation(df, *, root=DEFAULT_FIG_ROOT)` @ L236 — Summarise ablation grids when available.

## `src/synthetic/__init__.py`
- module: `synthetic`
- classes: none
- functions: none

## `src/synthetic/calibration.py`
- module: `synthetic.calibration`
- classes:
  - `CalibrationConfig` @ L26 — Configuration for synthetic calibration of overlay thresholds.
  - `ThresholdEntry` @ L48
  - `GridStat` @ L70
  - `CalibrationResult` @ L94
- functions:
  - `_simulate_panel(config, rng, *, spike_strength=0.0)` @ L122
  - `_select_entry(candidates, alpha)` @ L145
  - `calibrate_thresholds(config)` @ L163
  - `write_thresholds(result, path)` @ L244
  - `_run_seed_batches(*, config, seeds, spike_strength, edge_modes, delta_frac_values, stability_values)` @ L252
  - `_seed_batch_worker(payload)` @ L299

## `src/synthetic/threshold_eval.py`
- module: `synthetic.threshold_eval`
- classes:
  - `DetectionArrays` @ L15
- functions:
  - `_extract_detection_arrays(detections, *, alignment_min, require_isolated)` @ L28
  - `_evaluate_delta_grid(arrays, *, delta_abs, delta_frac_values)` @ L99
  - `evaluate_threshold_grid(observations, groups, *, delta_abs, eps, edge_modes, delta_frac_values, stability_values, q_max, a_grid, require_isolated=True, alignment_min=0.0, stats=None)` @ L127

## `src/utils/credentials.py`
- module: `utils.credentials`
- classes: none
- functions:
  - `_kc(service, account)` @ L3
  - `wrds_user()` @ L13
  - `wrds_password(user=None)` @ L16
  - `wrds_creds()` @ L20

## `tools/aggregate_runs.py`
- module: `tools.aggregate_runs`
- classes: none
- functions:
  - `_resolve_runs(patterns)` @ L14
  - `_load_run_metadata(run_dir)` @ L26
  - `aggregate_runs(run_dirs)` @ L48
  - `parse_args()` @ L79
  - `main()` @ L102

## `tools/build_brief.py`
- module: `tools.build_brief`
- classes: none
- functions:
  - `_format_percent(value)` @ L31
  - `_safe_float(value)` @ L37
  - `_aggregate_reason_table(reason_df)` @ L44
  - `build_brief(config_path)` @ L69
  - `parse_args()` @ L232
  - `main()` @ L243

## `tools/build_gallery.py`
- module: `tools.build_gallery`
- classes: none
- functions:
  - `_load_config(path)` @ L31
  - `_discover_run_paths(entries)` @ L37
  - `_gather_rejections(summary_df, run_tag)` @ L66
  - `_load_ablation(run_path)` @ L80
  - `_edge_dataframe(rolling_df, run_tag)` @ L91
  - `build_gallery(config_path)` @ L106
  - `parse_args()` @ L177
  - `main()` @ L188

## `tools/build_memo.py`
- module: `tools.build_memo`
- classes: none
- functions:
  - `_load_config(path)` @ L30
  - `_discover_run_paths(entries)` @ L35
  - `_latest_summary_dir(root=Path('reports'))` @ L64
  - `_load_summary_artifacts(config)` @ L74
  - `_format_kill_criteria_payload(kill_data)` @ L143
  - `_markdown_table(df)` @ L181
  - `_format_percent(value)` @ L200
  - `_pick_strategy_row(group, strategies)` @ L206
  - `_format_delta(value)` @ L217
  - `_format_ci(lo, hi)` @ L228
  - `_format_edge_metric(value)` @ L234
  - `_format_pvalue(value)` @ L242
  - `_format_detection(value)` @ L250
  - `_format_windows(value)` @ L256
  - `_numeric(row, key)` @ L262
  - `_row_for(frame, regime, portfolio)` @ L271
  - `_prettify_reason(reason)` @ L286
  - `_collect_rejection_records(summary_df, run_tag)` @ L290
  - `_build_key_tables(panel_df)` @ L301
  - `_build_rejection_tables(rejection_records)` @ L528
  - `build_memo(config_path)` @ L556
  - `parse_args()` @ L1297
  - `main()` @ L1308

## `tools/clean_outputs.py`
- module: `tools.clean_outputs`
- classes: none
- functions:
  - `is_tagged_directory(path)` @ L12
  - `unique_destination(base)` @ L16
  - `collect_legacy(root)` @ L29
  - `clean_outputs(root, *, purge, dry_run)` @ L40
  - `parse_args()` @ L78
  - `main()` @ L101

## `tools/generate_project_state.py`
- module: `tools.generate_project_state`
- classes: none
- functions:
  - `rel_path(path)` @ L58
  - `should_skip_dir(relative_dir)` @ L62
  - `categorize(path)` @ L81
  - `collect_files()` @ L103
  - `module_name_from_path(py_path)` @ L136
  - `get_py_files()` @ L148
  - `_format_default(node)` @ L172
  - `_format_signature(args)` @ L181
  - `parse_symbols(py_files)` @ L217
  - `extract_make_targets(makefile)` @ L304
  - `main()` @ L331

## `tools/list_runs.py`
- module: `tools.list_runs`
- classes:
  - `RunInfo` @ L35
- functions:
  - `_load_json(path)` @ L13
  - `_load_detection_total(path)` @ L23
  - `_extract_run_info(label, path)` @ L48
  - `discover_runs(base_dir)` @ L90
  - `format_runs(runs)` @ L107
  - `main()` @ L158

## `tools/make_summary.py`
- module: `tools.make_summary`
- classes:
  - `SummaryArtifacts` @ L20
- functions:
  - `_read_csv(path)` @ L27
  - `_normalise(series, value)` @ L36
  - `_pick_row(df, *, regime, estimator, portfolio)` @ L42
  - `_pick_dm_row(df, *, regime, portfolio, baseline='baseline')` @ L58
  - `_aggregate_row(df, *, regime, estimator, portfolio)` @ L72 — Aggregate matching rows (mean of numeric columns, first of non-numeric).
  - `_aggregate_dm_row(df, *, regime, portfolio, baseline='baseline')` @ L93
  - `_nan_median(series)` @ L111
  - `_nan_quantile(series, q)` @ L118
  - `_count_nonzero(series)` @ L126
  - `_concat_if_exists(paths)` @ L133
  - `_aggregate_diag_row(df)` @ L144
  - `_numeric(series, key)` @ L159
  - `_string(series, key, default='')` @ L168
  - `_load_detail(rc_dir, regime, root_detail)` @ L174
  - `_row_for(perf_df, regime, portfolio=None)` @ L188
  - `_criterion_entry(key, label, value, passed, threshold)` @ L200
  - `_evaluate_kill_criteria(perf_df, det_df, rc_run, regime='full')` @ L216
  - `summarise_rc_directory(rc_dir)` @ L319
  - `_discover_rc_dirs(root, patterns, all_runs, rc_dir)` @ L510
  - `_display_path(path)` @ L529
  - `write_summaries(rc_dirs)` @ L536
  - `parse_args(argv=None)` @ L596
  - `main(argv=None)` @ L624

## `tools/plot_rc_hist.py`
- module: `tools.plot_rc_hist`
- classes: none
- functions:
  - `_load_series(path, metric, regime)` @ L11
  - `plot_histogram(diagnostics_path, metric, out_path, *, title=None, bins=40, regime=None)` @ L20
  - `main()` @ L49

## `tools/prewhiten_effect.py`
- module: `tools.prewhiten_effect`
- classes:
  - `RunSummary` @ L53
- functions:
  - `_read_csv_rows(path)` @ L13
  - `_scalar(value)` @ L26
  - `_mode_from_resolved(run_dir)` @ L33
  - `_portfolio_value(rows, portfolio, column)` @ L62
  - `_sign_p_value(path, portfolio)` @ L73
  - `_load_run_summary(run_dir)` @ L85
  - `_build_effect_rows(off, on, label_off, label_on)` @ L117
  - `parse_args(argv=None)` @ L150
  - `main(argv=None)` @ L172

## `tools/reduce_calibration.py`
- module: `tools.reduce_calibration`
- classes: none
- functions:
  - `parse_args(argv=None)` @ L14
  - `main(argv=None)` @ L58

## `tools/run_monitor.py`
- module: `tools.run_monitor`
- classes:
  - `MetricSample` @ L44
- functions:
  - `_now_iso()` @ L32
  - `_safe_json(line)` @ L36
  - `_aggregate_process_metrics(proc)` @ L56
  - `_io_counters(proc)` @ L83
  - `_monitor_loop(proc, interval, metrics_path, progress_queue, stop_event, hostname, samples)` @ L107
  - `_summarise(samples)` @ L215
  - `main()` @ L239

## `tools/shard_grid.py`
- module: `tools.shard_grid`
- classes: none
- functions:
  - `parse_args(argv=None)` @ L14
  - `_shard_jobs(jobs, shard_count, strategy)` @ L62
  - `main(argv=None)` @ L79

## `tools/summarize_rc_sanity.py`
- module: `tools.summarize_rc_sanity`
- classes: none
- functions:
  - `_delta_mse(metrics, portfolio)` @ L29
  - `_effect_label(delta_ew, delta_mv)` @ L40
  - `_load_daily_payload(path, label)` @ L55 — Best-effort loader for daily diagnostics/metrics.
  - `_load_weekly_payload(path)` @ L110 — Best-effort loader for weekly summary + detection.
  - `_merge_completeness(entry, comp)` @ L155
  - `_build_daily_entry(label, path)` @ L172
  - `_build_weekly_entry(label, path)` @ L185
  - `_aggregate_entries(entries)` @ L194
  - `main()` @ L214

## `tools/summarize_run.py`
- module: `tools.summarize_run`
- classes: none
- functions:
  - `_read_json(path)` @ L15
  - `_safe_read_csv(path)` @ L25
  - `_fmt_float(x)` @ L34
  - `summarize_run(output_dir)` @ L40
  - `main()` @ L303

## `tools/summarize_weekly_diagnostics.py`
- module: `tools.summarize_weekly_diagnostics`
- classes: none
- functions:
  - `_reason_series(df)` @ L11
  - `_format_reason_summary(df, top_k)` @ L19
  - `_render_stat_table(df, columns)` @ L34
  - `_guardrail_totals(df)` @ L48
  - `_format_reason_examples(df, top_k, example_k=5)` @ L58
  - `summarize(input_path, output_path, top_k=5)` @ L141
  - `main()` @ L184

## `tools/update_registry.py`
- module: `tools.update_registry`
- classes: none
- functions:
  - `compute_sha256(path)` @ L15
  - `summarise_returns(path)` @ L23
  - `load_registry(path)` @ L33
  - `main()` @ L43

## `tools/verify_dataset.py`
- module: `tools.verify_dataset`
- classes: none
- functions:
  - `_sha256(path)` @ L11
  - `_normalise_key(path)` @ L19
  - `_load_registry(path)` @ L27
  - `verify_dataset(dataset_path, registry_path)` @ L38
  - `main()` @ L64
