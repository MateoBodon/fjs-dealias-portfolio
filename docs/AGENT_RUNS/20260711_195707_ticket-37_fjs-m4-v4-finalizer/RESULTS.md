# Results

The v4 finalizer now has a complete fail-closed lifecycle:

- one required month and one deterministic cell identity for every month from
  `2013-01` through `2018-12`;
- a generation-bound receipt for each cell artifact, primary source, all
  lookback-source hashes, and shared factor binding;
- exact restart idempotence without permitting a different artifact to replace
  a completed month;
- atomic checkpoint writes and independent reload validation;
- finalization only at exactly 72 unique, ordered months and cells;
- a source catalog that rejects missing, duplicate, conflicting, outside-range,
  or mutated sources;
- aggregate source-set, cell-set, and manifest hashes;
- independent final readback of source, factor, cell, and manifest bytes.

The final manifest schema can certify realistic-design input completeness only.
It remains `full_execution_ready=false`, `aws_execution_authorized=false`, and
`outcomes_present=false`; 2025 and the legacy ticker CSV remain unopened and
unused.

The focused synthetic proof constructs all 72 required sources/cells and
passes byte-stable finalization/readback. It is contract evidence, not the real
2013-2018 derivation or detector evidence.
