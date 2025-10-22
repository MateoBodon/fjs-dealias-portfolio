# fjs-dealias-portfolio

De-aliasing the spurious spikes that arise when MANOVA spectra are aliased in high-dimensional regimes yields materially better out-of-sample covariance and risk forecasts than Ledoit–Wolf shrinkage, enabling more reliable portfolio design under market noise.

## Quickstart

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `make setup`
4. `make test`

## Experiments

- Synthetic MANOVA stress test: `make run-synth`
- Equity panel replication using historical prices: `make run-equity`
- Adjust the YAML files under `experiments/*/config.yaml` to explore alternate scenarios.

## Citation

Fan, J., Johnstone, I. M., & Sun, Q. (2018). Eigenvalue shrinkage estimation of large covariance matrices. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*.
