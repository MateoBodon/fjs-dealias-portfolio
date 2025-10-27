# Methods: Balanced MANOVA De-aliasing for Risk Forecasting

This document summarizes the core methodology implemented in this repository and maps code symbols to the notation used in the FJS framework for balanced designs.

## Balanced one-way MANOVA and weekly risk

We work with the balanced one-way MANOVA design used in FJS (2018) with

- I groups indexed by week,
- J replicates per group indexed by business day,
- observation matrix \(Y \in \mathbb{R}^{(I\cdot J)\times p}\) ordered by week/day.

The random-effects decomposition assumes
\[
r_{ij} = \mu_i + \varepsilon_{ij},\qquad
\mathbb{E}[\mu_i] = \mathbf{0},\quad \mathbb{E}[\varepsilon_{ij}] = \mathbf{0},
\]
with independent \(\mu_i\) across weeks, independent \(\varepsilon_{ij}\) across days, and \(\operatorname{Cov}(\mu_i) = \Sigma_1,\ \operatorname{Cov}(\varepsilon_{ij}) = \Sigma_2\). Replicates are conditionally independent given the week effect, so cross-covariances vanish when \(j \ne j'\).

Balanced MANOVA yields the mean squares (code: `src/fjs/balanced.py`) and component estimators
\[
\widehat{\Sigma}_2 = \widehat{\text{MS}}_2,\qquad
\widehat{\Sigma}_1 = \frac{\widehat{\text{MS}}_1 - \widehat{\text{MS}}_2}{J},
\]
corresponding to the design weights \(c_1 = J\) and \(c_2 = 1\).

Define the weekly aggregate for portfolio weights \(w\) as
\[
R_i = \sum_{j=1}^J r_{ij},\qquad v_i = w^\top R_i.
\]
Using the assumptions above,
\[
\begin{aligned}
\operatorname{Cov}(R_i)
&= \sum_{j=1}^J \sum_{j'=1}^J \operatorname{Cov}(r_{ij}, r_{ij'}) \\
&= \sum_{j=1}^J \operatorname{Cov}(r_{ij}, r_{ij}) + \sum_{j\ne j'} \operatorname{Cov}(r_{ij}, r_{ij'}) \\
&= J\,\Sigma_2 + J(J-1)\,\Sigma_1 + J\,\Sigma_1 \\
&= J^2 \Sigma_1 + J \Sigma_2.
\end{aligned}
\]
Therefore the weekly risk of the aggregate return satisfies
\[
\operatorname{Var}(v_i) = \operatorname{Var}\!\left[\sum_{j=1}^J w^\top r_{ij}\right]
= J^2 w^\top \widehat{\Sigma}_1 w + J\, w^\top \widehat{\Sigma}_2 w,
\]
and both the aliased and de-aliased estimators must target the same \(\widehat{\Sigma}_1,\widehat{\Sigma}_2\) components before and after spike substitution.

## Balanced nested Year⊃Week design

The nested configuration introduces a hierarchical factor: calendar years contain iso-weeks, which in turn contain balanced daily replicates. After dropping imperfect (year, week) cells and intersecting weeks across retained years, the observation cube is indexed as

- `i = 1,…,I` (years),
- `j = 1,…,J` (weeks per retained year),
- `r = 1,…,R` (daily replicates inside each week).

With random effects \(\mu_i \sim (0,\Sigma_1)\), \(\nu_{ij} \sim (0,\Sigma_2)\), and \(\varepsilon_{ijr} \sim (0,\Sigma_3)\) mutually independent, the balanced mean squares implemented in `src/fjs/balanced_nested.py` are
\[
\begin{aligned}
\widehat{\text{MS}}_1 &= \frac{R J}{I-1} \sum_{i=1}^I (\bar{Y}_{i..} - \bar{Y}_{...})(\bar{Y}_{i..} - \bar{Y}_{...})^\top,\\
\widehat{\text{MS}}_2 &= \frac{R}{I(J-1)} \sum_{i=1}^I \sum_{j=1}^J (\bar{Y}_{ij.} - \bar{Y}_{i..})(\bar{Y}_{ij.} - \bar{Y}_{i..})^\top,\\
\widehat{\text{MS}}_3 &= \frac{1}{IJ(R-1)} \sum_{i,j,r} (Y_{ijr} - \bar{Y}_{ij.})(Y_{ijr} - \bar{Y}_{ij.})^\top.
\end{aligned}
\]
The component estimators generalize to
\[
\widehat{\Sigma}_3 = \widehat{\text{MS}}_3,\qquad
\widehat{\Sigma}_2 = \frac{\widehat{\text{MS}}_2 - \widehat{\text{MS}}_3}{R},\qquad
\widehat{\Sigma}_1 = \frac{\widehat{\text{MS}}_1 - \widehat{\text{MS}}_2}{R J}.
\]
The detector consumes the accompanying metadata (`d = [I-1, I(J-1), IJ(R-1)]`, `c = [JR, R, 1]`, `N = R`, order sets `[[1,2,3],[2,3],[3]]`) so the Marchenko–Pastur mapping can switch between one-way and nested regimes without modifying the core MP routines.

## Marchenko–Pastur surrogate z(m) and bulk edge

For balanced designs, the surrogate transform z(m) admits a closed form depending on design weights a, c, degrees d, the sample size proxy N, and trace‑based plug‑ins C_s (denoted Cs in code). We implement:

- `z_of_m(m, a, C, d, N, Cs)` and its derivatives (code: `src/fjs/mp.py`).
- The admissible real root m(λ) of z(m)=λ with positive slope z′(m)>0 (`admissible_m_from_lambda`).
- The upper edge z₊ located at a stationary point with negative curvature (z′=0, z″<0) (`mp_edge`).

Mapping to code symbols:

- a ↔ `a` (grid over S¹ for one‑way),
- c ↔ `C` (design coefficients),
- d ↔ `d` (degrees per stratum),
- N ↔ `N` (replicate count J for one‑way),
- C_s ↔ `Cs` (trace plug‑ins estimated from mean squares with top‑eigen trimming).

## t‑vector and spike acceptance (Algorithm 1)

For an outlier candidate λ̂ (eigenvalue of Σ̂(a) above the MP edge), we compute the t‑vector t(λ̂, a) with components aggregated by the order relation between strata. For one‑way (two strata), the acceptance criteria target component r=1 (between‑group):

1) Edge and buffer: λ̂ ≥ z₊(a) + δ, or λ̂ ≥ z₊(a)·(1+δ_frac) when a relative buffer is used.
2) Component dominance (t‑vector): |t_r| ≥ ε and max_{s≠r} |t_s| ≤ ε.
3) Angular stability: acceptance persists under a → a(θ±η) within a small neighborhood.

If accepted, the de‑aliased spike estimate is μ̂ = λ̂ / t_r. We then substitute μ̂ along the detected direction into Σ̂₁.

Code:

- `t_vec(λ, a, C, d, N, c, order, Cs)` in `src/fjs/mp.py`.
- Algorithm 1 search and guardrails in `src/fjs/dealias.py` (`dealias_search`).
- The detector persists per-candidate diagnostics (`edge_margin = λ̂ - z₊`, |t| vector, admissible root check with z₀'(m) > 0) for inspection in `rolling_results.csv` and `summary.json`.

### θ root finding for k = 2

For the one-way design (`k=2`) we optionally refine the coarse angular grid by solving `t₂(λ̂, θ) = 0`, i.e. forcing the off-component of the aggregated t vector to vanish. The helper `solve_theta_for_t2_zero` (`src/fjs/theta_solver.py`) scans a coarse grid for a sign change, applies a Brent-style bisection, and rechecks the guardrails at `θ±δ` before accepting the root. The CLI flag `--oneway-a-solver {auto,rootfind,grid}` controls this behaviour: `auto` tries the root finder and falls back to the grid when no stable bracket exists, `rootfind` insists on a root (falling back to grid when necessary), and `grid` preserves the legacy search. Each accepted detection now reports the solver used in the JSON diagnostics.

## Noise plug‑ins and trimming

We estimate Cs from mean squares using eigenvalue trimming to reduce the impact of large spikes:

- `estimate_Cs_from_MS([MS1, MS2], d, c, drop_top=k)`, where `k` can be controlled by a fraction `cs_drop_top_frac` of the feature dimension p.

## Weekly covariance reconstruction and OOS risk

Given Σ̂₁, Σ̂₂ and accepted spikes { (μ̂_ℓ, v̂_ℓ) }, we form

1) Adjusted Σ̂₁ by substituting Rayleigh quotients along v̂_ℓ with μ̂_ℓ for the top‑K directions.
2) Weekly covariance Σ̂_weekly = J²·Σ̂₁(adj) + J·Σ̂₂.
3) Forecast portfolio variance wᵀ Σ̂_weekly w and compare to realized weekly sums on the holdout.

Baselines now include aliased (no substitution), Ledoit–Wolf, OAS shrinkage-to-identity, constant-correlation shrinkage, a simple SCM fallback, and the observed-factor covariance that regresses asset returns on provided factor portfolios (optionally with industry returns). All baselines operate on the identical weekly panel so risk forecasts remain comparable across methods.

The minimum-variance portfolio uses a projected-gradient solver with ridge regularisation (`minvar_ridge_box`) and configurable box bounds (`--minvar-box`). Turnover is tracked window by window; when `--turnover-cost` is supplied the realised variance is debited by the one-way transaction cost before computing squared errors. Both the ridge and the turnover adjustments are reflected in `rolling_results.csv` and the per-window cache signature.

Diebold–Mariano statistics complement the paired sign test: for each strategy we test de-aliased squared errors against each baseline (Ledoit–Wolf, OAS, constant-correlation, factor) using a Newey–West long-run variance estimate. The metrics summary (`metrics_summary.csv`) exposes the t-statistic and p-value columns (`dm_stat_*`, `dm_p_*`) so runs can be compared at a glance.

## Defaults and recommended knobs (equity)

- δ_frac ≈ 0.03, ε ≈ 0.03, η ≈ 0.4°, a_grid ≈ 144 (conservative), signed a enabled.
- Tuned demo: δ_frac≈0.02, ε≈0.02, η≈0.2°, a_grid≈180.
- Crisis windows can increase detection likelihood; use `--crisis YYYY-MM-DD:YYYY-MM-DD`.

## Validation

- Synthetic S1–S5:
  - S1: spectrum with MP edge and outliers.
  - S2: eigenvector alignment with planted spike.
  - S3: bias μ̂ vs aliased λ̂ across signal strengths (μ ∈ {4,6,8}).
  - S4: guardrail FPR ≤ 1% under isotropy; lax settings yield high FPR.
  - S5: multi‑spike behavior and pairing (naive vs alignment‑based).

- Equity:
  - E1: weekly covariance spectrum with MP edge overlay.
  - E2: aliased λ̂ vs de‑aliased μ̂ over windows (or alt: top weekly eigenvalue series when no detections).
  - E3: OOS variance MSE across methods.
  - E4: 95% VaR coverage error.
  - E5: ablations over δ_frac/ε/a_grid/η.
