# Results

## Deterministic detector milestone

- Production now uses `N=p-L`, the correct one-way inclusion lattice, and a
  single explicit-`C_s` equation (5.5) map across edge, root, derivative, and
  component calculations.
- Reachable one-way theta roots are enumerated and seeded from the four
  strongest above-edge outliers before candidate gating.
- Spectral reconstruction uses symmetric multi-candidate subspace replacement,
  preserves the orthogonal block, is sign/permutation invariant, and fails
  loudly on rank-deficient candidate sets.
- The three former strict expected failures are ordinary passing tests. The
  frozen independent oracle was not edited.
- The historical Ticket 24 total-injection artifact remains an off-target
  negative control and still fails the target-between provenance requirement.

## Frozen fixture

- Predeclaration commit: `82d1ffc0b2fc7c4c39e820b7aae3c4ad0bcdb43c`.
- Source commit/tree: `9afbb72cb02172080c52ba206ddd73ed2110dedf` /
  `75ce510b7193eee457f825fa57d6417a2c361170`.
- Frozen result: null detection/acceptance `0/12`; `mu=6`, target-between
  detection/acceptance `12/12`.
- `input_spec.json` SHA-256:
  `e212e430e91bbd285eb20b11090f08b31a7c929ec8208ea1da541b57e94c5093`.
- `curve.csv` SHA-256:
  `d19edfb7bdfa22fab487a1e0ff551bc346435340fa467fd4c02f77c446848a07`.
- `trials.csv` SHA-256:
  `0620d728f509bed0a3ae8f065f22e2330a1c696c2918a16638a34ab6cc076f7f`.
- Generator SHA-256:
  `632ad21f54bdca4447b65173dd07d8c2107ff4d66b418b340e5b0e2fe587a38e`.
- Byte-for-byte reproduction: PASS.

## Claim boundary and continuation

This is a two-cell synthetic mechanism calibration and deterministic detector
correctness result. It is not a full size/power study, a real-data empirical
result, or promotion evidence.

The next fresh goal is preparation only: freeze a full null/power/invariance
cell manifest and checkpoint/restart runner suitable for a separately
authorized AWS `c7i.16xlarge` execution. It must use target-between injection,
exact-binomial cell gates, direction/component attribution, invariance checks,
deterministic seeds, process orchestration, and no tuning. This run did not
implement or launch that package.

Substantive result checkpoint:

- commit: `143b74972392a0fa1ba0ece6384a33d2c1a663fd`
- tree: `8315f975e24ef5c387c5b58350ce578097eb9098`
- parent at milestone start: `e6df52575a1db65c85372ddaa545ae62698ee507`

The final evidence commit and exact-tree remote readback are reported to the
Portfolio Administrator. No routine handoff bundle is produced.
