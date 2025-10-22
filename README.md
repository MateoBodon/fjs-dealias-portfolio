# fjs-dealias-portfolio

De-aliasing the spurious spikes that arise when MANOVA spectra are aliased in high-dimensional regimes yields materially better out-of-sample covariance and risk forecasts than Ledoit–Wolf shrinkage, enabling more reliable portfolio design under market noise.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make fmt
make lint
make test
```

## Running the experiments

- **Synthetic (S1/S3)** – `make run-synth`
  - Outputs: `figures/synthetic/s1_histogram.(png|pdf)`, `figures/synthetic/bias_table.csv`, `figures/synthetic/summary.json`.
- **Equity rolling forecast** – `make run-equity`
  - Outputs: `experiments/equity_panel/outputs/` (rolling CSV summaries and variance/VaR figures).

Both scripts accept optional `--config` YAML files mirroring the defaults in the corresponding `experiments/*/config.yaml`.

## Reproducing figures

Running the commands above regenerates all main figures and tables in the `figures/` and `experiments/equity_panel/outputs/` directories. The synthetic pipeline emits S1 spectra histograms and S3 bias tables; the equity experiment writes variance/VaR comparison plots alongside CSV logs of every rolling window.

## De-aliasing heuristics

The de-aliasing search employs a *delta buffer* (\(\delta\)) that requires candidate eigenvalues to exceed the Marčenko–Pastur bulk edge by a safety margin before being treated as spikes. A *stability check* perturbs the search direction by ±η degrees and only accepts spikes whose classification is invariant to this perturbation, preventing brittle detections caused by numerical noise.

## Citation

Fan, J., Johnstone, I. M., & Sun, Q. (2018). Eigenvalue shrinkage estimation of large covariance matrices. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*.

## Citation

Fan, J., Johnstone, I. M., & Sun, Q. (2018). Eigenvalue shrinkage estimation of large covariance matrices. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*.
