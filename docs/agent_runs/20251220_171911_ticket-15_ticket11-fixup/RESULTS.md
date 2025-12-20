Current HEAD:
b932a0d6ace045508f372afb76284e0c04f03b1a

Git status --short:
?? docs/agent_runs/20251220_171911_ticket-15_ticket11-fixup/

Uncommitted eval-contamination-looking changes: none (only run log directory)

Smoke eval output (CAPPED — not for headline summaries)
Out dir: reports/ticket-15-smoke-171911

full/dm.csv (header + first 3 rows):
portfolio,baseline,dm_stat,p_value,n_effective,dm_stat_qlike,p_value_qlike,n_effective_qlike,comparison_valid,comparison_valid_qlike
ew,baseline,,,0,,,0,0,0
ew,lw,,,0,,,0,0,0
ew,oas,,,0,,,0,0,0

full/metrics.csv (header + first 3 rows):
regime,estimator,portfolio,mse_mean,es_mse_mean,var95,es95,realised_var,realised_es,violation_rate,count,qlike_mean,mv_turnover_mean,mv_turnover_cost_bps,cov_condition_median,cov_condition_p90,baseline_mse,baseline_es_mse,baseline_qlike_mean,delta_mse_vs_baseline,delta_es_vs_baseline,delta_qlike_vs_baseline,delta_mse_ci_lower,delta_mse_ci_upper,n_effective_mse,n_effective_es,n_effective_qlike,comparison_valid_mse,comparison_valid_es,comparison_valid_qlike,comparison_valid
full,baseline,ew,2.7362097821976636e-09,,-0.015718914318431126,-0.019712152712842157,4.401170546381797e-05,,0.0,5,0.3142907682351549,0.0,0.0,6.483460263037003,6.93823902875329,2.7362097821976636e-09,,0.3142907682351549,,,,,,,,,,,,
full,baseline,mv,1.4443091758910329e-09,,-0.013644937757475138,-0.017111302433730765,3.3339180055867426e-05,,0.0,5,0.2843236789024743,0.015410600923834044,0.0,6.483460263037003,6.93823902875329,1.4443091758910329e-09,,0.2843236789024743,,,,,,,,,,,,
full,cc,ew,2.247056207440098e-08,,-0.022776591288103,-0.02856276436487936,4.401170546381797e-05,,0.0,5,0.7990874090396092,0.0,0.0,42.7105555430624,44.82289776231041,2.7362097821976636e-09,,0.3142907682351549,,,,,,,,,,,,

skip_stats.csv (header + first 3 rows):
regime,portfolio,estimator,skip_reason,skip_count,windows,skip_share
calm,ew,baseline,,0,5,0.0
calm,ew,cc,,0,5,0.0
calm,ew,ewma,,0,5,0.0

run.json windows block:
{
  "cap_active": true,
  "cap_sources": [
    "max_windows",
    "window_coverage"
  ],
  "window_coverage": 0.0013408420488066506,
  "windows_after_caps": 5,
  "windows_evaluated": 5,
  "windows_requested": 3729
}

Overlay row excerpts (full):
regime,estimator,portfolio,delta_mse_vs_baseline,delta_es_vs_baseline,delta_qlike_vs_baseline,n_effective_mse,n_effective_es,n_effective_qlike,comparison_valid
full,overlay,ew,0.0,,0.0,5.0,0.0,5.0,0.0
portfolio,baseline,dm_stat,p_value,n_effective,dm_stat_qlike,p_value_qlike,n_effective_qlike,comparison_valid,comparison_valid_qlike
ew,baseline,,,0,,,0,0,0


Smoke eval output (rerun after final code changes — CAPPED — not for headline summaries)
Out dir: reports/ticket-15-smoke-171911

full/dm.csv (header + first 3 rows):
portfolio,baseline,dm_stat,p_value,n_effective,dm_stat_qlike,p_value_qlike,n_effective_qlike,comparison_valid,comparison_valid_qlike
ew,baseline,,,0,,,0,0,0
ew,lw,,,0,,,0,0,0
ew,oas,,,0,,,0,0,0

full/metrics.csv (header + first 3 rows):
regime,estimator,portfolio,mse_mean,es_mse_mean,var95,es95,realised_var,realised_es,violation_rate,count,qlike_mean,mv_turnover_mean,mv_turnover_cost_bps,cov_condition_median,cov_condition_p90,baseline_mse,baseline_es_mse,baseline_qlike_mean,delta_mse_vs_baseline,delta_es_vs_baseline,delta_qlike_vs_baseline,delta_mse_ci_lower,delta_mse_ci_upper,n_effective_mse,n_effective_es,n_effective_qlike,comparison_valid_mse,comparison_valid_es,comparison_valid_qlike,comparison_valid
full,baseline,ew,2.7362097821976636e-09,,-0.015718914318431126,-0.019712152712842157,4.401170546381797e-05,,0.0,5,0.3142907682351549,0.0,0.0,6.483460263037003,6.93823902875329,2.7362097821976636e-09,,0.3142907682351549,,,,,,,,,,,,
full,baseline,mv,1.4443091758910329e-09,,-0.013644937757475138,-0.017111302433730765,3.3339180055867426e-05,,0.0,5,0.2843236789024743,0.015410600923834044,0.0,6.483460263037003,6.93823902875329,1.4443091758910329e-09,,0.2843236789024743,,,,,,,,,,,,
full,cc,ew,2.247056207440098e-08,,-0.022776591288103,-0.02856276436487936,4.401170546381797e-05,,0.0,5,0.7990874090396092,0.0,0.0,42.7105555430624,44.82289776231041,2.7362097821976636e-09,,0.3142907682351549,,,,,,,,,,,,

skip_stats.csv (header + first 3 rows):
regime,portfolio,estimator,skip_reason,skip_count,windows,skip_share
calm,ew,baseline,,0,5,0.0
calm,ew,cc,,0,5,0.0
calm,ew,ewma,,0,5,0.0

run.json windows block:
{
  "cap_active": true,
  "cap_sources": [
    "max_windows",
    "window_coverage"
  ],
  "window_coverage": 0.0013408420488066506,
  "windows_after_caps": 5,
  "windows_evaluated": 5,
  "windows_requested": 3729
}

Overlay row excerpts (full):
regime,estimator,portfolio,delta_mse_vs_baseline,delta_es_vs_baseline,delta_qlike_vs_baseline,n_effective_mse,n_effective_es,n_effective_qlike,comparison_valid
full,overlay,ew,0.0,,0.0,5.0,0.0,5.0,0.0
portfolio,baseline,dm_stat,p_value,n_effective,dm_stat_qlike,p_value_qlike,n_effective_qlike,comparison_valid,comparison_valid_qlike
ew,baseline,,,0,,,0,0,0

Bundle path:
- docs/gpt_bundles/20251220_174554_ticket-15_20251220_171911_ticket-15_ticket11-fixup.zip
- Contents: docs/agent_runs/20251220_171911_ticket-15_ticket11-fixup/bundle_contents.txt

