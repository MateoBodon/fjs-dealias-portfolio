# RC Memo — full

## ΔMSE Summary
portfolio            ew            mv
estimator                            
ewma       3.831241e-09 -4.856673e-10
overlay    7.186814e-07  8.412839e-07
rie        0.000000e+00  0.000000e+00
sample    -1.002000e-11 -5.759864e-10

- **EW** best ΔMSE: sample (-1.0020e-11)
- **MV** best ΔMSE: sample (-5.7599e-10)

## Diebold–Mariano Tests
portfolio baseline  dm_stat  p_value
       ew      rie      NaN      NaN
       mv      rie      NaN      NaN

## VaR Coverage
portfolio        ew        mv
estimator                    
ewma       0.238095  0.333333
overlay    0.000000  0.000000
rie        0.238095  0.404762
sample     0.238095  0.333333

## Postmortem
- Null FPR stress (p=60, μσ₂=8, 20 trials) dropped from 100% to 0% after enabling isolation + off-component caps (see `reports/rc-20251102/calibration/thresholds.json`).
- Power at μσ₁=6 exceeds 98% over 80 trials in `tests/fjs/test_dealias_calibrated.py`.
- Detection defaults: δ=0.5, δ_frac=0.0, t_eps=0.06, off_component_cap=0.3, require_isolated=True, q_max=1.
