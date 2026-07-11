# Tests

- Focused repaired v4 contract suite — PASS (`6 passed`).
- Combined v3/v4/finalizer suite after migration support — PASS (`81 passed`).
- Prior 15 monthly cell cores regenerated exactly — PASS
  (`2def914231a3cddc0b914806dcafa9f46261cccb0fde0129f2a4f7a791a35458`).
- Exact checkpoint migration revalidated 15/15 retained artifacts — PASS.
- Full checkpoint completeness — PASS (`72/72`, no missing month).
- Independent manifest readback — PASS; two readback files were byte-identical.
- Aggregate source, cell, and manifest digest checks — PASS.
- Explicit detector outcome, AWS, and 2025 false boundaries — PASS.
- Aggregate geometry QA — FAIL for flagship adequacy by design: zero
  missingness and only 9-13 residual dates per cell.

No raw source, cell, checkpoint, full manifest, or detector outcome was added
to Git.
