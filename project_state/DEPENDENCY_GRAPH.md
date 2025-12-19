---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Dependency Graph

Source: project_state/_generated/import_graph.json (internal imports only).

**Top fan-out (modules importing many peers)**

module                         | fan_out | imports (truncated)                                                                                             
------------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------
equity_panel.run               | 21      | baselines, evaluation, evaluation.evaluate, experiments.prewhiten, finance.eval, finance.io…                    
eval.run                       | 16      | baselines, baselines.covariance, baselines.factors, eval.balance, eval.clean, evaluation.dm…                    
finance.eval                   | 7       | evaluation.factor, finance.factors, finance.ledoit, finance.robust, finance.shrinkage, fjs.balanced…            
eval.inject_spike              | 6       | eval.balance, eval.clean, experiments.daily.grouping, experiments.eval.config, experiments.eval.run, fjs.overlay
synthetic_oneway.run           | 6       | fjs.balanced, fjs.dealias, fjs.spectra, meta.run_meta, pairing, plotting                                        
fjs.overlay                    | 5       | baselines.covariance, finance.ledoit, finance.shrinkage, fjs.dealias, fjs.gating                                
test_pipeline_smoke            | 5       | experiments.equity_panel, experiments.synthetic_oneway, finance, fjs, fjs.dealias                               
synthetic.power_null           | 5       | evaluation.evaluate, experiments.synthetic_oneway.run, finance.eval, fjs.dealias, fjs.robust                    
synthetic.nested_killtest      | 5       | experiments.equity_panel.run, fjs.balanced_nested, fjs.dealias, fjs.gating, fjs.robust                          
test_threshold_eval            | 4       | fjs.balanced, fjs.overlay, synthetic.calibration, synthetic.threshold_eval                                      
test_dealias_search            | 4       | finance.io, fjs.balanced, fjs.dealias, fjs.mp                                                                   
synthetic.calibrate_thresholds | 4       | experiments.synthetic.harness_utils, fjs, meta, synthetic.calibration                                           
fjs.dealias                    | 3       | fjs.balanced, fjs.mp, fjs.theta_solver                                                                          
build_gallery                  | 3       | report.gather, report.plots, report.tables                                                                      
test_dealias_guardrails        | 3       | fjs.balanced, fjs.dealias, fjs.mp                                                                               
test_diagnostics               | 3       | experiments.equity_panel, io, tools.summarize_run                                                               
test_shrinkage                 | 3       | finance.ledoit, finance.robust, finance.shrinkage                                                               
test_dealias                   | 3       | fjs.balanced, fjs.dealias, fjs.mp                                                                               
test_theta_solver              | 3       | fjs.balanced, fjs.dealias, fjs.theta_solver                                                                     
experiments.test_eval_run      | 3       | experiments.eval.config, experiments.eval.diagnostics, experiments.eval.run                                     

**Top fan-in (modules that many peers import)**

module                              | fan_in
----------------------------------- | ------
fjs.dealias                         | 17    
fjs.balanced                        | 12    
fjs.mp                              | 8     
evaluation.evaluate                 | 8     
experiments.equity_panel            | 8     
report.gather                       | 6     
experiments.eval.run                | 6     
experiments.synthetic.harness_utils | 5     
finance.shrinkage                   | 5     
fjs.gating                          | 5     
fjs.robust                          | 5     
evaluation.dm                       | 4     
finance.ledoit                      | 4     
finance.io                          | 4     
baselines.covariance                | 4     
fjs.overlay                         | 4     
finance.factors                     | 3     
evaluation.factor                   | 3     
finance.robust                      | 3     
fjs.spectra                         | 3     

**Notes**
- Modules are named from file paths (src/ stripped). Relative imports were resolved best-effort.
- rc/evaluation entrypoints (experiments/equity_panel/run.py, experiments/eval/run.py) dominate fan-out; core math modules (fjs.*) are high fan-in.