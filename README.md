# fjs-dealias-portfolio

De-aliasing the spurious spikes that arise when MANOVA spectra are aliased in high-dimensional regimes yields materially better out-of-sample covariance and risk forecasts than Ledoit–Wolf shrinkage, enabling more reliable portfolio design under market noise. In a balanced one-way design with $J$ daily replicates per week, the weekly risk of a portfolio with weights $w$ decomposes into

$$
\mathbb{V}\!\left[\sum_{j=1}^J w^\top r_j\right] = J^2 w^\top \widehat{\Sigma}_1 w + J\, w^\top \widehat{\Sigma}_2 w,
$$

highlighting why both the aliased and de-aliased estimators must target the same $\widehat{\Sigma}_1, \widehat{\Sigma}_2$ components even when we correct the spike magnitudes.

## Quickstart

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install project and tooling:
   ```bash
   make setup
   ```
3. Regenerate the synthetic evidence suite (S1/S2/S3/S4/S5):
   ```bash
   make run-synth
   ```
4. Reproduce the rolling equity experiment:
   ```bash
   make run-equity
   ```

`make test` remains available to run the full pytest suite; `make fmt` / `make lint` apply formatting and static checks.

## Testing

- Fast feedback (skips long statistical tests):
  - `make test-fast` (equivalent to `pytest -m "not slow" -n auto` with fallback to serial if xdist is missing)
- Full suite with parallel workers:
  - `make test-all` (equivalent to `pytest -n auto`, also falls back to serial)
- Only the slow group (for targeted checks):
  - `pytest -m slow -n auto`

Optional fast mode for slow tests
- Set `FAST_TESTS=1` to reduce angle grid sizes (`a_grid`) and trial counts in the heaviest property-style tests while preserving their intent. Without `FAST_TESTS`, slow tests run at their original strict settings.
- Examples:
  - `FAST_TESTS=1 make test-all`
  - `FAST_TESTS=1 pytest -m slow -n auto`

Parallelism
- Parallel execution is provided by `pytest-xdist` (installed by `make setup`).
- `-n auto` uses all logical cores; on laptops you can set a fixed worker count, e.g., `-n 6` to limit thermal/CPU load.

Notes
- `cvxpy` loads lazily in portfolio optimizers; if not present, the code falls back to equal-weight behavior where appropriate, and tests remain valid.

## Methods at a glance

- **Balanced MANOVA decomposition:** weekly returns are partitioned into between-group ($\widehat{\Sigma}_1$) and within-group ($\widehat{\Sigma}_2$) mean squares using `fjs.balanced.mean_squares`.
- **t-vector acceptance:** spikes are accepted only when the Marchenko–Pastur t-vector has dominant support on the target component, ensuring $\hat{\mu} = \hat{\lambda} / t_r$ remains self-consistent. A relative δ buffer `dealias_delta_frac` can be used to scale the MP edge decision by a fraction of the edge.
- **Guardrails:** candidates must clear an MP edge buffer (`δ`), survive angular perturbations (`η`), and win cluster merges based on stability margin.
- **Risk forecasting:** detected spikes are substituted into $\widehat{\Sigma}_1$ before recombining weekly covariance for equal-weight and box-constrained min-variance portfolios; Ledoit–Wolf provides the shrinkage baseline.

## Configuration & CLI flags

| Option | Default | Description |
| --- | --- | --- |
| `dealias_delta` | `0.3` | Safety buffer added to the MP edge before accepting eigenvalue outliers. |
| `dealias_eps` | `0.05` | Minimum absolute t-vector mass for the target component and tolerance for off-target entries. |
| `stability_eta_deg` | `1.0` | Angular perturbation (in degrees) applied when checking directional stability. |
| `dealias_delta_frac` | `None` | Relative δ buffer as a fraction of the MP edge (e.g., 0.05 → 5%). Overrides `dealias_delta` when set. |
| `signed_a` | `True` | Search both positive and negative `a` directions (recommended for equity). |
| `cs_drop_top_frac` | `0.1` | Fraction of top eigenvalues dropped when estimating Cs from mean squares. |
| `--sigma-ablation` | `False` | When passed to `experiments/equity_panel/run.py`, perturbs empirical Cs by ±10% and records detection robustness. |
| `--crisis "YYYY-MM-DD:YYYY-MM-DD"` | `None` | Restrict the equity run to a crisis window; results are written to `outputs/crisis_*`. |
| `--config path/to/config.yaml` | — | Override defaults for data paths, horizons, or delta/eps/eta settings. |

The equity configuration file (`experiments/equity_panel/config.yaml`) mirrors these keys; adding `dealias_delta`, `dealias_delta_frac`, `signed_a`, `cs_drop_top_frac`, `dealias_eps`, or `stability_eta_deg` entries will override the defaults above.

## Figure gallery

- **Synthetic suite:**  
  `figures/synthetic/s1_histogram.(png|pdf)` – spectrum of $\widehat{\Sigma}_1$.  
  `figures/synthetic/s2_vectors.(png|pdf)` – alignment of the recovered eigvector with the planted spike.  
  `figures/synthetic/s4_guardrails.(csv|png|pdf)` – false-positive rates with and without guardrails.  
  `figures/synthetic/s5_multispike.(csv|png|pdf)` – aliased vs de-aliased bias across multiple spikes.  
  `figures/synthetic/bias_table.csv`, `figures/synthetic/summary.json` – tabulated S1–S5 metrics.

- **Equity panel:**  
  `experiments/equity_panel/outputs/spectrum.(png|pdf)` – weekly covariance spectrum.  
  `experiments/equity_panel/outputs/E3_variance_mse.(png|pdf)` – variance forecast MSE comparison.  
  `experiments/equity_panel/outputs/E4_var95_coverage_error.(png|pdf)` – VaR coverage error bars.  
  `experiments/equity_panel/outputs/variance_forecasts.png`, `var95_forecasts.png` – forecast time‑series overlays (baseline vs de‑aliased).  
  `experiments/equity_panel/outputs/rolling_results.csv`, `metrics_summary.csv`, `summary.json` – per‑window diagnostics (detections, forecasts, realised risk).

Running `make run-synth` and `make run-equity` is sufficient to refresh the full gallery.

### Embedded previews

Note: The paths below point to locally generated artefacts. We now publish the fast synthetic previews (figures/synthetic_fast) so they render on GitHub. Equity outputs remain local by default unless explicitly committed.

#### Synthetic suite (fast)

S1 — Spectrum of \(\widehat{\Sigma}_1\)

![S1 histogram](figures/synthetic_fast/s1_histogram.png)

S2 — Leading eigenvector vs. planted spike

![S2 vectors](figures/synthetic_fast/s2_vectors.png)

S4 — Guardrail false-positive comparison

![S4 guardrails](figures/synthetic_fast/s4_guardrails.png)

S5 — Multi-spike bias (aliased vs. de-aliased)

![S5 multispike](figures/synthetic_fast/s5_multispike.png)

### Fast Synthetic Results (Summary)

- S1 spectrum: Clear outlier above MP edge; the empirical noise median and edge indicate a well-separated spike consistent with the planted signal.
- S2 alignment: Leading eigvector aligns strongly with the planted direction (cosine ≈ 0.92 in a representative run).
- S3 bias (µ=6): Aliased top eigenvalue shows large positive bias; de-aliased estimate reduces bias to near the true magnitude with ~100% detection in the fast run.
- S4 guardrails: Default guardrails suppress false positives on isotropic data (≈0% FPR), while lax settings admit many spurious detections—highlighting the guardrails’ importance.
- S5 multi-spike: With two planted spikes, simple top-k pairing by λ̂ can yield higher de-aliased bias than aliased. This suggests refining pairing (e.g., Rayleigh alignment) or tightening acceptance (smaller eps, larger a_grid) for interacting spikes.

#### Equity panel

E1 — Weekly covariance spectrum (fit window)

![Equity spectrum](experiments/equity_panel/outputs/spectrum.png)

E3 — Variance forecast MSE

![E3 variance MSE](experiments/equity_panel/outputs/E3_variance_mse.png)

E4 — 95% VaR coverage error

![E4 VaR coverage error](experiments/equity_panel/outputs/E4_var95_coverage_error.png)

<!-- E5 ablation visuals are not generated in this repo version; omitted from preview. -->

Rolling overlays — variance and VaR forecasts (baseline vs. de-aliased)

![Variance forecasts](experiments/equity_panel/outputs/variance_forecasts.png)

![VaR95 forecasts](experiments/equity_panel/outputs/var95_forecasts.png)

## Matching targets across estimators

Both the aliased estimator and the de-aliased spike reconstructions are calibrated against the same weekly covariance components. We first compute the balanced MANOVA mean squares \(\widehat{\text{MS}}_1, \widehat{\text{MS}}_2\), then build the weekly covariance through

$$
\widehat{\Sigma}_\text{weekly} = J^2 \widehat{\Sigma}_1 + J \widehat{\Sigma}_2,\qquad \widehat{\Sigma}_1 = \frac{\widehat{\text{MS}}_1 - \widehat{\text{MS}}_2}{J},\quad \widehat{\Sigma}_2 = \widehat{\text{MS}}_2.
$$

De-aliasing only substitutes selected spike magnitudes in $\widehat{\Sigma}_1$; $\widehat{\Sigma}_2$ and the aggregation to weekly risk remain unchanged, guaranteeing both paths forecast the same quantity before and after spike adjustments. Ledoit–Wolf operates on the same balanced weekly returns, providing a shrinkage baseline against the identical target.

## Guardrails during de-aliasing

- **δ-buffer:** candidate spikes must exceed the Marčenko–Pastur bulk edge plus a safety buffer before they are considered.
- **Angular stability:** every accepted spike must persist when the search direction $a$ is rotated by ±η degrees.
- **Cluster merge:** detections with nearby $\hat{\mu}$ values are merged; the most stable representative is kept.

## Recommended defaults and CLI flags

- Recommended equity defaults: `dealias_delta=0.0`, `dealias_delta_frac=0.03`, `dealias_eps=0.03`, `stability_eta_deg=0.4`, `cs_drop_top_frac=0.05`, `signed_a=true`, `a_grid=144`.
- Equity CLI flags: `--delta-frac`, `--eps`, `--a-grid`, `--eta`, `--sigma-ablation`, `--ablations`, `--crisis`, `--no-progress`.

See `METHODS.md` for a compact technical summary of Algorithm 1, acceptance criteria, and the weekly aggregation identity.

## Recommended equity knobs

- Conservative (fewer false positives): `dealias_delta_frac=0.03`, `dealias_eps=0.03`, `stability_eta_deg=0.4`, `a_grid=144`, `signed_a=true`.
- Tuned demo (more detections): `--delta-frac 0.02 --eps 0.02 --eta 0.2 --a-grid 180 --signed-a`.
- Crisis subperiods often reveal outliers; use `--crisis "YYYY-MM-DD:YYYY-MM-DD"` (e.g., early 2020).

## Baselines

Alongside Aliased/De-aliased, we compare Ledoit–Wolf and (when applicable) sample covariance (SCM). The equity runner emits all methods’ errors and coverage.

## Citation

Fan, J., Johnstone, I. M., & Sun, Q. (2018). Eigenvalue shrinkage estimation of large covariance matrices. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*.
