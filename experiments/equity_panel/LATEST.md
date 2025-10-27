# Equity Panel Outputs

The canonical artefacts for the most recent full equity run live in
`experiments/equity_panel/outputs`. Older slices (smoke tests, crisis zoom-ins,
etc.) are parked under `experiments/equity_panel/archive/<timestamp_label>/` to
keep the top-level directory tidy while preserving provenance. The latest run
committed to this repo is `experiments/equity_panel/archive/2025-10-27_full`.

## How to inspect available runs

```bash
python tools/list_runs.py
```

The helper prints each run’s label, sample period, number of rolling windows,
and the key guardrail settings (`off_component_leak_cap`, `energy_min_abs`,
`delta_frac`, `signed_a`). Point the `--base` flag at an alternative directory if
you maintain a separate workspace.

## Refreshing the current run

1. Update `experiments/equity_panel/config.yaml` with any new guardrail settings.
2. Run `make run-equity` (≈90 minutes locally). The command rewrites
   `experiments/equity_panel/outputs` in place.
3. If you want to retain the previous results, move the old `outputs` directory
   into `archive/` **before** launching the regeneration:

   ```bash
   mv experiments/equity_panel/outputs \
      experiments/equity_panel/archive/$(date +%Y-%m-%d)_full
   ```

   After the rerun finishes, the directory tree remains clean and the new
   artefacts stay easy to find.
