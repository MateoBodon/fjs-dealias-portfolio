# FJS Scientific Recenter Predeclaration

status: frozen before detector repair and broad empirical execution
frozen_at: 2026-07-10
ticket: 37
base_commit: `193a325dc681ebc4da67b44715a92e4f63113019`
decision_owner: Portfolio Administrator

## Decision and research question

The project will pursue the ambitious result only through a falsifiable sequence:

> Can the Fan-Johnstone-Sun (FJS) high-dimensional random-effects detector
> identify planted covariance components in realistic balanced equity panels,
> and, after separating FJS identification from generic Marcenko-Pastur
> substitution, improve point-in-time covariance forecasts over a frozen set of
> modern shrinkage, robust, factor, and dynamic baselines?

The current daily result is not evidence for this claim. Detector validation is
the stop-line. No broad CRSP run, advisor-facing performance claim, or tuning to
the confirmation/holdout period is allowed until the detector gates below pass.

## Frozen claim hierarchy

1. **Positive result:** the FJS-only arm clears every detector gate, and its
   primary loss improves over the best frozen baseline on paired confirmation
   windows with a one-sided 95% confidence bound below zero. Portfolio risk
   does not worsen after turnover costs.
2. **Conditional result:** detector validity passes but FJS helps only in a
   predeclared VIX or aspect-ratio stratum. Report that boundary without
   relabeling it as an unconditional effect.
3. **Negative result:** detector validity passes but FJS does not beat the best
   baseline, or the FJS arm is too rare for the minimum effective sample. This
   is a bounded negative performance result.
4. **Mechanism failure / pivot:** detector validity fails. Stop performance
   tuning and repair or reject the implemented FJS mapping. Generic coarse,
   oracle, and sham arms may diagnose the mechanism but cannot rescue an FJS
   claim.

## Evidence audit at freeze

- The recovered Ticket 24 weekly injection curve is exactly flat at zero for
  detection and acceptance at `mu` in `{0, 3, 6, 12, 24}`. Its most common
  pre-gate failures are off-component and no-root reasons. The small canonical
  reference and its hashes are in
  `docs/artifacts/detector-contract-reference/ticket24_week_full_fix/`.
- The recovered T-012 matrix is useful historical evidence but not a clean FJS
  validation. All 6,917 changed full-regime windows across its four legs are
  attributed to the generic `coarse` fallback; the FJS-only contribution is
  therefore not identified. Across the eight leg-by-portfolio comparisons, the
  overlay beats the repo's crude RIE comparator on QLIKE but loses to the best
  implemented CC/EWMA comparator each time. EW MSE worsens in all four legs.
  The curated source is `docs/artifacts/rc-t-012/`; its recovery caveat remains.
- The current `data/returns_daily.csv` has SHA-256
  `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`,
  892,529 rows, 300 ticker labels, and dates 2010-01-05 through 2024-12-31.
  It lacks PERMNO/security filters and previously selected the first ticker
  labels alphabetically when `assets_top` was set. It is not eligible for the
  headline universe.
- The repo's current `rie` implementation is convex shrinkage toward the mean,
  and its `quest` label is Marcenko-Pastur clipping. Neither may be described as
  authoritative analytical nonlinear shrinkage or QuEST in the flagship
  comparison. Missing/failed factor and POET outputs cannot be silently dropped.

## Candidate-source contract

Every candidate must carry exactly one fail-loud `candidate_source` label:

| Label | Meaning | Claim use |
|---|---|---|
| `fjs` | Root/component returned by the FJS de-aliasing map | Only arm eligible for an FJS claim |
| `coarse` | Generic spectral candidate from the coarse fallback | Diagnostic comparator only |
| `oracle` | Planted direction/component supplied by mechanism calibration | Upper-bound diagnostic only |
| `sham` | Magnitude-matched non-FJS direction | Negative-control diagnostic only |

Missing, unknown, or mixed labels fail. A coarse arm may not be pooled with an
FJS arm, and a baseline implementation name may not be silently redirected to a
different estimator. Reports must preserve pre-gate and accepted source counts.

## Detector stop-line

The following gates are frozen before further detector changes. Synthetic or
semi-synthetic data are mechanism calibration only, never the headline result.

1. **Reference equivalence:** deterministic low-dimensional balanced one-way
   cases reproduce an independently computed reference for the MP edge, roots,
   mapped component, and reconstructed covariance within declared numerical
   tolerances. Permuting assets or groups must preserve eigenvalues and map the
   direction accordingly.
2. **Null size:** at the frozen nominal 5% operating point, the empirical FJS
   detection rate's exact 95% binomial interval must contain 5%, and its upper
   bound must be at most 7.5%. Acceptance cannot exceed detection. Results must
   be reported for every frozen `(p, T, groups, replicates)` cell, not only a
   pooled average.
3. **Planted power:** at a spike 1.5 times the independently computed detection
   boundary, FJS detection and acceptance must each be at least 80%, detection
   must improve by at least 50 percentage points over the null, and the frozen
   power curve must be nondecreasing. Squared cosine to the planted direction
   must be at least 0.80, at least 90% of accepted candidates must map to the
   planted component, and nuisance-component attribution must be at most 10%.
4. **Invariance and provenance:** standardized rescaling, deterministic row
   order, asset permutation, and group-label permutation must not change the
   decision beyond numerical tolerance. No coarse/oracle/sham candidate may
   appear in the FJS arm, including after numerical failure.
5. **Real-design adequacy:** repeat the null and planted tests using the exact
   missingness, aspect ratios, group sizes, and residual covariance structures
   sampled from development CRSP windows. The same thresholds apply; calibration
   cannot tune on confirmation or holdout outcomes.
6. **Attribution sufficiency:** before a performance claim, the FJS-only arm
   must change the estimate on at least 30 non-overlapping confirmation dates.
   Fewer changes yield a rare/absent mechanism conclusion, not a pooled coarse
   result.
7. **Promotion gate:** on common paired confirmation windows, the one-sided 95%
   confidence bound for the primary loss difference versus the best frozen
   baseline must be below zero. Gross minimum-variance risk and net risk after
   the frozen 5 bps turnover charge may not worsen. Any failure stops promotion.

`src/fjs/detector_contract.py` implements the persisted power-curve reducer for
the null-rate, strong-power, acceptance, gain, and monotonicity subset. The
remaining gates require the next bounded reference harness. The strict expected
failure against the historical curve deliberately keeps the current blocker
visible in the unit suite.

## Real-data design and provenance

### Frozen authorities

- CRSP 2010-2017: `/Volumes/Storage/Data/wrds/_manifests/20260707T214900Z_worker8_crsp_dsfv2_month_2017_2010_csvgz/manifest.json`
  (`status=ok`, 96 items, zero failures).
- CRSP 2018-2023: `/Volumes/Storage/Data/wrds/_manifests/20260707T204600Z_worker7_crsp_dsfv2_month_recent_csvgz/manifest.json`
  (`status=ok`, 72 items, zero failures).
- Corresponding raw partitions:
  `/Volumes/Storage/Data/WRDS/raw/crsp/wrds_dsfv2_query/snapshot=20260707_045553_global_project_priority/month=YYYY-MM/data.csv.gz`.
- CRSP 2024 and 2025 candidate receipts:
  `/Volumes/Storage/Data/wrds/_manifests/20260707_full_p1_core_fast_csvgz/manifest.json`,
  with item-level `ok` receipts for
  `/Volumes/Storage/Data/WRDS/raw/crsp/dsf_v2/snapshot=20260707_0005_fast_partitioned/year=2024/data.csv.gz`
  (2,403,637 rows; 199,721,045 bytes) and `year=2025/data.csv.gz`
  (2,541,801 rows; 210,784,573 bytes). The enclosing manifest has unrelated
  failures, so dedicated content-hashed derived manifests are mandatory before
  either year is used.
- Fama-French 5-factor archive SHA-256:
  `bcf32ecc9e2bb20383784ac98891e42146a0091eec6ec77d3b5bf0d4e981e3f6`.
- Momentum archive SHA-256:
  `f4237e2e36dffa13fd7823f55376316a94b5ac663af951dd9eaca8ed2c678bcf`.
- Current combined factor file SHA-256:
  `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`.
- CBOE VIX:
  `/Volumes/Storage/Data/Public/public_cboe_vix_vol_indexes/daily_prices/VIX_History.csv`,
  1990-01-02 through 2026-07-08, SHA-256
  `3a909bc8987edd6b6c08873a09abcc74c9697aa1b93004d6725475e21e7164b6`.
- Exchange calendars:
  `/Volumes/Storage/Data/Public/public_exchange_corporate_event_calendars/generated/XNYS_sessions_1990_2035.csv`
  SHA-256 `29b6e84fe9072a67ae0f88bb046a98d89e6945cb496263ed5386793724060deb`
  and `XNAS_sessions_1990_2035.csv` SHA-256
  `479ee9d59f62619352903d6f44e929f5374a844c0b6ad9a01b57219795488367`.

Restricted raw data remain outside Git. Every derived result must bind the
exact input manifest, file hash, extraction date, filters, universe members,
ranking date, and code commit. No synthetic substitute is admissible.

### Point-in-time universe

Security identity is PERMNO. On each formation date, rank only information
available before the forecast starts. The intended CRSP CIZ filters are:
`securitytype=EQTY`, `securitysubtype=COM`, `sharetype=NS`, `usincflg=Y`,
primary exchange in `{N, A, Q}`, conditional type `RW`, and trading status `A`.
Require price at least $5, adequate trailing-return history with no future
backfill, and rank by lagged market capitalization. The primary universe is the
top 60 eligible names, reconstituted point in time with membership recorded.

The daily runner now refuses `assets_top` unless an explicit ranked snapshot
CSV (`as_of_date,ticker,rank`) and as-of date are supplied; it hashes and records
the source and selected symbols. This is a safe static smoke mechanism, not the
headline implementation. The flagship run additionally requires a rolling
point-in-time universe adapter with PERMNO and the filters above.

### Windows and data splits

- Primary residualization: FF5 plus momentum (FF6); raw returns are a declared
  robustness arm. Factor fitting uses only past observations.
- A complete estimation block contains 156 weeks, grouped by week with five
  weekday replicates where the exchange calendar is complete. The target
  component is fixed at component 0. The forecast horizon is four weeks and
  evaluation dates are non-overlapping for primary inference.
- Warm-up/construction: 2010-2012. Development and detector repair: 2013-2018.
  Confirmation: 2019-2023. The year 2024 is an exposed bridge/robustness period.
  Calendar 2025 is the only true final holdout and stays unopened until all
  code, thresholds, baselines, and the claim reducer are frozen.
- VIX strata are fixed from lagged VIX information before the window and are
  used only for predeclared conditional reporting, never outcome-driven slicing.

## Frozen baseline ladder

All baselines run on identical point-in-time inputs and paired forecast dates.
Failures are reported, never dropped or replaced.

1. Sample covariance and diagonal/equal-correlation controls; equal-weight
   portfolio as a decision baseline.
2. Ledoit-Wolf linear shrinkage, OAS, and constant-correlation shrinkage.
3. A genuine, independently checked analytical nonlinear shrinkage/QuEST
   implementation; the current MP clipper and crude RIE remain labeled legacy
   diagnostics until replaced.
4. Robust nonlinear shrinkage (R-NL) and a Tyler-based robust comparator.
5. FF6 factor covariance, POET, and a weak-factor-aware SAF comparator.
6. EWMA and a documented large dynamic covariance baseline.
7. Separate `fjs`, `coarse`, `oracle`, and magnitude-matched `sham` arms.

## Endpoints, reducer, and holdout policy

The primary endpoint is the multivariate Gaussian quasi-log score on the
four-week realized covariance, computed on common dates. Secondary endpoints
are gross and 5-bps-turnover-adjusted minimum-variance realized variance, EW
QLIKE, condition number/stability, concentration, and turnover. Results are
paired; a method failure removes that date from every method in the comparison
and is separately counted. No unpaired sample-size advantage is permitted.

The reducer evaluates, in order: data/provenance completeness; detector gates;
FJS-only attribution count; baseline completeness; paired primary confidence
bound; portfolio non-inferiority; then predeclared robustness. A coarse-only
gain, an oracle gain, an exposed-period gain, or a secondary endpoint cannot
override a failed earlier gate. Tuning after seeing 2025 invalidates the holdout
and must be disclosed as a new development generation.

## Execution stages

1. Reproduce the historical flat-zero artifact by hash and keep its strict
   expected failure visible. **Current stage.**
2. Build the independent deterministic reference harness and diagnose the
   root/component mapping without CRSP-scale execution.
3. Freeze and pass the full null/power/invariance/real-design detector suite.
4. Build the rolling CRSP adapter and dedicated 2024/2025 derived manifests;
   validate point-in-time membership on a bounded sample.
5. Implement and independently validate every frozen baseline.
6. Run development, then confirmation once. Reduce without changing the gate.
7. Only if confirmation promotes the design, open and run the 2025 holdout once.

The immediate next command is the bounded deterministic reference-harness test;
a full CRSP or memory-heavy run remains prohibited while the historical curve
fails the detector stop-line.

## Literature frozen for implementation review

- Fan, Johnstone, and Sun, *Spiked covariances and principal components analysis
  in high-dimensional random effects models*: <https://arxiv.org/abs/1806.09529>
- Ledoit and Wolf, *Nonlinear shrinkage estimation of large-dimensional
  covariance matrices*: <https://arxiv.org/abs/1207.5322>
- Fan, Liao, and Mincheva, *Large covariance estimation by thresholding
  principal orthogonal complements (POET)*: <https://arxiv.org/abs/1201.0175>
- Robust nonlinear shrinkage (R-NL): <https://arxiv.org/abs/2210.14854>
- Large dynamic covariance estimation:
  <https://www.econ.uzh.ch/dam/jcr%3A28fa9939-753e-4f5c-932d-945872f30cfd/jbes_2019.pdf>
- SAF weak-factor covariance estimation:
  <https://academic.oup.com/jfec/article/23/1/nbae017/7725018>
