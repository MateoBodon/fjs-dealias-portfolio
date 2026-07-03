# Prompt / Task Summary

- ticket: T-012
- phase: execute
- label: [none]

## Summary

Execute T-012 end-to-end inside the locked daily `dow` empirical-only lane:
create the two tracked `window252` configs, run the monitored four-leg
`window x prewhiten` matrix under `reports/rc-t-012/` with at most two live
eval legs at a time, preserve partial trees instead of silently rerunning,
build the shared summary pack plus `t012_full_regime_comparison.csv` and
`campaign_decision.md`, update the required state docs, validate the run log,
and build a self-describing GPT bundle.
