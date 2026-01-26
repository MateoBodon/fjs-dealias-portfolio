# History cleanup plan (no filter-repo run yet)

## Findings (largest blobs)
- data/wrds/returns_daily.parquet.xz.partaa (~90 MiB)
- data/wrds/returns_daily.parquet.xz.partab (~34 MiB)
- data/prices_sample.csv (~26 MiB)
- reports/aws/.../risk.csv and diagnostics_detail.csv (multi‑MB)
- reports/runs/.../metrics.jsonl (sub‑MB)

## Proposed cleanup (plan only)
1. Confirm whether the large `data/` and `reports/` blobs are still required in git history; if not, list them in an allowlist for removal.
2. Coordinate a history rewrite window and notify collaborators to avoid force‑push surprises.
3. Run `git filter-repo` (or BFG) on a dedicated branch to remove the large blobs and any old run dumps under `reports/` or `data/` that violate `TRACKING_POLICY.md`.
4. Rebuild repo size metrics (`git count-objects -vH`) and verify no large blobs remain.
5. Add/confirm ignore rules to prevent reintroducing large files (already in `.gitignore` for `data/**/*.parquet`, `reports/_runs/`).
6. Force-push the rewritten history and rotate clones (fresh clone or `git fetch --all` + `git reset --hard`).

## Notes
- This plan does **not** execute history rewrite; it is a proposal only.
- If any large blobs must remain, migrate them to Git LFS with explicit approval.
