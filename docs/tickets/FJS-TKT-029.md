# FJS-TKT-029

## Goal
Make gpt_bundle audit-correct by failing on dirty repos so DIFF.patch always matches the ticket work.

## Scope
- Update `tools/agentic/gpt_bundle.py` to enforce clean git status and propagate git_dirty into BUNDLE_META.
- Add regression coverage in `tests/test_gpt_bundle.py`.
- Add a minimal docs note about git_dirty in bundle metadata.
- Do not touch eval logic or inject_spike math.

## Acceptance Criteria
- If `git status --porcelain` is non-empty, gpt_bundle exits non-zero with a clear message.
- `BUNDLE_META.md` includes `git_dirty=true/false`.
- Regression test added in `tests/test_gpt_bundle.py`.
- `make test-fast` and `pytest -q tests/test_gpt_bundle.py` pass.

## Plan
1. Inspect gpt_bundle wrapper and current bundle metadata behavior.
2. Add clean-repo guard and bundle-meta update in `tools/agentic/gpt_bundle.py`.
3. Extend `tests/test_gpt_bundle.py` with dirty-check/meta tests.
4. Add minimal docs note on `git_dirty` in BUNDLE_META.
5. Run required tests, generate bundle, and update logs.

## Notes
- Keep diffs minimal and local to bundling workflow.
