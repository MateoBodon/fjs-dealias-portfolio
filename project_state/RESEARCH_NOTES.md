# Research Notes

## Problem framing
- Balanced one-way and nested MANOVA designs over equity-return panels (week/day, year/week/day). Goal: detect between-group spikes (Σ₁) aliased into pooled covariance, then de-alias to improve portfolio risk forecasts.
- Observations ordered by groups (weeks or years→weeks) with replicates = trading days.

## Key notation ↔ code
- Mean squares (MS): `fjs.balanced.mean_squares`, `balanced_nested.mean_squares_nested` compute MS₁ (between), MS₂ (within), (MS₃ nested residual) and component estimators Σ̂₁, Σ̂₂ (, Σ̂₃).
- Design weights: `c` (e.g., [J,1] one-way; [JR,R,1] nested), degrees `d` (I−1, …), replicates `N` = J or R.
- MP transform z(m), derivatives z′, z″: `fjs.mp.z_of_m`, `z0_prime`, `z0_double_prime`; edge z₊ = `mp_edge` (stationary point with negative curvature).
- Admissible root m(λ) and t-vector: `fjs.mp.admissible_m_from_lambda`, `t_vec` with order sets (one-way [[1,2],[2]]; nested [[1,2,3],[2,3],[3]]).
- Spike acceptance (Algorithm 1, `fjs.dealias.dealias_search`):
  1) λ̂ above edge + buffer δ or δ_frac·z₊.
  2) Target component |t_r| ≥ ε and off-components ≤ ε (optional off-component ratio cap).
  3) Stability under small angular perturbations a(θ±η) (edge margin stays ≥0).
  4) Optional energy floor and admissible m(λ̂) root.
  5) Optional θ root finder (k=2) to refine a where t₂=0.
- De-aliased eigenvalue μ̂ = λ̂ / t_r substituted into Σ̂₁ along eigenvector v̂.

## Robust edges
- Scatter alternatives (`fjs.robust.tyler_scatter`, `huber_scatter`) scale MP edge via `edge_from_scatter` and `edge_mode` in overlay/experiments (scm/tyler/huber). Edge scaling recorded in detections.

## Overlay & gating
- `fjs.overlay.detect_spikes` couples detection with calibrated δ_frac lookups (`fjs.gating.lookup_calibrated_delta`), isolation requirement, stability/alignment thresholds, soft top-k gating, q_max cap, optional coarse candidates (simple MP edge check from sample covariance).
- Overlay substitution uses shrinkage/factor baselines (RIE, LW, OAS, CC, factor_obs, POET-lite, Tyler-shrink, EWMA) before eigenvalue replacement.

## Evaluation metrics
- Weekly risk reconstruction: Σ_weekly = J²Σ̂₁(adj) + JΣ̂₂ (`finance.eval.weekly_cov_from_components`).
- Forecast loss: ΔMSE vs baselines on EW and min-var portfolios; VaR/ES (95%) coverage tests (Kupiec/Christoffersen) and ES t-test; DM and sign tests (`evaluation.dm`, `evaluation.evaluate`). Flip-set DM focuses on windows where overlay changed forecasts.
- Alignment diagnostics: angle between detection direction and top-k PCA components (`evaluation.evaluate.alignment_diagnostics`).

## Conceptual notes / potential mismatches
- Balanced assumption is strict; partial weeks are dropped or imputed—guardrails may over-restrict nested runs when sample sizes thin.
- Cs estimation drops top eigenvalues; `scan_basis` sigma vs ms changes scaling of MP inputs (auto α based on Σ̂ mean). Might bias thresholds if spectra highly skewed.
- Overlay enforces PSD via eigenvalue clipping/ridge; strong clipping could mask deleterious detections—metrics currently cap by min forecast vs aliased baseline in `variance_forecast_from_components` (keeps min of overlay and 0.9×baseline when detections present).
- rc-lite-sanity shows moderate detection (~5%) but overlay touches ~94–100% of windows, yielding positive ΔMSE and failing kill criteria—suggests coarse-candidate + gate-soft combination may over-apply substitutions once any detection exists.
- Daily summary aggregates only DoW; vol-state run remains outside summary, obscuring cross-design comparison.
- Weekly smoke/nested runs still emit zero detections at tyler edges despite relaxed guardrails and calibrated δ—may stem from balanced-week filtering or still-too-tight t-vector/edge buffers for p≈188, T≈60–80.
