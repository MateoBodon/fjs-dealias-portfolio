# FJS Deterministic Reference Harness Findings

status: deterministic production reference gate passed
verified_at: 2026-07-10  
ticket: 37  
scope: balanced one-way detector correctness; no empirical performance claim

## Decision

Broad experiments remain blocked, but the bounded production correctness repair
is complete. The independent reference implementation defines and tests the
minimum balanced one-way FJS contract, and production now satisfies that exact
contract without changing the oracle or reducer thresholds.

Run the stop-line with:

```bash
make detector-reference-gate
```

The command now exits zero against the manifest-bound between-component
mechanism fixture. The historical Ticket 24 curve still fails if supplied
explicitly, as it must, because its target-treatment provenance is absent.

## Independent reference

`src/fjs/reference_oracle.py` is deliberately independent of production
`fjs.mp`, `fjs.dealias`, and `fjs.overlay`. It implements FJS equation (5.5),
enumerates polynomial roots instead of reusing production brackets, maps an
isolated component through the explicit balanced-design inclusion lattice, and
reconstructs an eigenpair as

`(I-vv') B (I-vv') + mu vv'`.

The frozen two-stratum reference uses `I=5`, `J=2`, `p=5`, one planted spike,
and therefore `N=p-L=4`. Its exact inputs and checked values are:

| Quantity | Frozen value |
|---|---:|
| `a` | `[21/sqrt(2041), -40/sqrt(2041)]` |
| `C` | `[3, 1]` |
| `d` | `[4, 5]` |
| component inclusion | group `[[1]]`; residual `[[1,2]]` in paper indexing |
| admissible `m` | `-sqrt(2041)/168` |
| sample outlier `lambda` | `1176/(5 sqrt(2041)) = 5.206139668687108` |
| upper edge | `4.871008798276616` |
| `t` | `[336/(5 sqrt(2041)), 0]` |
| mapped `mu` | `3.5` |

A scalar special case independently checks edge `4.5`, admissible root `-0.4`
at `lambda=5`, `t=1.25`, and mapped `mu=4`. Homogeneity, stratum permutation,
direction-sign invariance, and orthogonal-block preservation are also tested.

Oracle and sham controls are deterministic and source-labeled. The oracle uses
the normalized planted direction. The sham uses the first stable basis residual
that is orthogonal to the oracle, with identical `mu` and `lambda`; it is a
mechanism control, never an empirical result.

## Historical Ticket 24 evidence boundary

The hash-bound Ticket 24 source commit is
`31c05a57ffd5db7a1531c427eb7373de5f7a5f22`. At that revision,
`experiments/eval/inject_spike.py` standardized one iid observation-level series
and added its outer product to the full panel. The recorded command in
`docs/agent_runs/20251226_060917_ticket-24_finish-week-inject-spike/COMMANDS.md`
did not select a component injection mode, and the recovered curve contains no
`inject_mode` field. The detector targets component zero.

Therefore the flat-zero curve at `mu` in `{0,3,6,12,24}` is valid evidence for
that exact total/observation-level configuration and a useful negative control.
It is not a target-between-component planted-power test. The new reducer fails
loudly when a target-power claim omits `inject_mode=between` or supplies another
mode.

## Resolved production failures

Production checkpoint `4437571acf4b42bd1f4c7db8a9616b623c5a3a7b`
resolved the four code failures, and subsequent bounded root-seeding commits
preserved the same outputs while restoring native-suite runtime:

1. `oneway_bulk_dimension_mismatch`: production uses the replicate count for
   `N`; the reference contract requires the bulk dimension `N=p-L`.
2. `oneway_inclusion_order_mismatch`: production reverses the one-way inclusion
   lattice. The group component uses the group stratum; residual noise uses both
   group and residual strata.
3. `explicit_cs_mp_map_mismatch`: the explicit-`C_s` MP edge/root/t path does
   not implement a single consistent equation (5.5) map.
4. `spectral_reconstruction_mismatch`: the overlay changes only the candidate's
   Rayleigh quotient when the direction is not already a baseline eigenvector;
   it does not install the claimed eigenpair.
5. `target_power_provenance_invalid`: the historical curve remains ineligible,
   while the new fixture carries exact `inject_mode=between` provenance.


The former strict expected failures are now ordinary passing production-to-
oracle equivalence tests. Missing or mismatched target-treatment provenance is
still rejected by contract.

## Frozen between-component mechanism fixture

The fixture was predeclared at commit
`82d1ffc0b2fc7c4c39e820b7aae3c4ad0bcdb43c` before generator implementation or
execution. Its fixed 12 paired trials used seed `20260710`, `I=60`, `J=3`,
`p=10`, within-noise scale `0.3`, and `mu` in `{0,6}`. The final production
replay is bound to source commit `9afbb72cb02172080c52ba206ddd73ed2110dedf`.

- null: detection `0/12`, acceptance `0/12`;
- `mu=6`, `inject_mode=between`: detection `12/12`, acceptance `12/12`;
- curve SHA-256:
  `d19edfb7bdfa22fab487a1e0ff551bc346435340fa467fd4c02f77c446848a07`;
- trial SHA-256:
  `0620d728f509bed0a3ae8f065f22e2330a1c696c2918a16638a34ab6cc076f7f`.

`tools/generate_fjs_between_fixture.py --check` reproduced both output hashes.
This establishes only deterministic mechanism plumbing at two cells. It is not
an exact-binomial null-size study, a full power curve, realistic-design evidence,
or an empirical result.

## Completed repair and next gate

1. One-way `N=p-L`, inclusion order, and canonical mean-square `C_s` semantics
   are corrected.
2. Edge, admissible root, and `t` use one exact equation (5.5) map and the same
   explicit `C_s`; rank-one theta roots are seeded from the strongest bounded
   outliers before candidate gating.
3. Multi-candidate reconstruction uses symmetric orthonormalisation and
   `PBP + Q diag(mu) Q'`, is permutation/sign invariant, and fails on deficient
   candidate spans.
4. `make detector-reference-gate` passes with the independent oracle unchanged.
5. The next gate is the frozen full null/power/invariance calibration with exact
   binomial cell reducers, direction/component attribution, and checkpointed
   deterministic execution. It remains mechanism calibration only.

No synthetic, semi-synthetic, or reference-harness output in this milestone is
a headline empirical result.
