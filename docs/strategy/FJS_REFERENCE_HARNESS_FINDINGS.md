# FJS Deterministic Reference Harness Findings

status: production repair blocked by five fail-loud reference-gate issues  
verified_at: 2026-07-10  
ticket: 37  
scope: balanced one-way detector correctness; no empirical performance claim

## Decision

Broad experiments remain blocked. The independent reference implementation now
defines and tests the minimum correctness contract for the balanced one-way FJS
path, but the production implementation does not yet satisfy it.

Run the stop-line with:

```bash
make detector-reference-gate
```

The command must exit non-zero while any issue below is present. A future repair
is eligible for null and planted-power calibration only when the command exits
zero without weakening the reference values or treatment-provenance check.

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

## Production blockers

The deterministic gate currently reports exactly these blockers:

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
5. `target_power_provenance_invalid`: the historical curve cannot support a
   target-between-component power claim because injection-mode provenance is
   absent.

Three strict expected failures keep the first four code-level mismatches visible
inside the unit suite. The fifth is an ordinary passing rejection test: missing
or mismatched target-treatment provenance is rejected by contract.

## Repair order and next gate

1. Correct one-way `N`, inclusion order, and canonical mean-square-to-`C_s`
   semantics.
2. Use one equation (5.5) implementation and identical explicit `C_s` values
   for edge, root, and `t` calculations; make the root solver reachable before a
   grid candidate has already passed the same gates.
3. Define and implement the intended multi-candidate reconstruction semantics.
4. Make `make detector-reference-gate` pass without changing the oracle.
5. Produce a bounded deterministic `inject_mode=between` calibration with exact
   manifests and only then run the frozen null/power/invariance suite.

No synthetic, semi-synthetic, or reference-harness output in this milestone is
a headline empirical result.
