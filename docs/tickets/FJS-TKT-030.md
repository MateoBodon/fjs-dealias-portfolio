# FJS-TKT-030

## Goal
Align gpt-bundle output + docs with TRACKING_POLICY by moving bundle zips into canonical scratch (`artifacts/_local/`) and removing `docs/gpt_bundles` as an output target.

## Scope
- Update the Makefile `gpt-bundle` recipe so bundles land under `artifacts/_local/gpt_bundles/`.
- Update `docs/DOCS_AND_LOGGING_SYSTEM.md` to reflect the canonical output zones.
- Update `tests/test_gpt_bundle.py` expectations for the new bundle path.
- Make `tools/agentic/gpt_bundle.py` safe on dirty trees (stash/restore) with a `--no-stash` override.
- Update `tools/gpt_bundle.py` only if necessary.

## Acceptance Criteria
- `make gpt-bundle` prints a `.zip` path under `artifacts/_local/gpt_bundles/`.
- Running `make gpt-bundle` does not dirty the repo (ignoring scratch).
- `docs/DOCS_AND_LOGGING_SYSTEM.md` no longer lists `docs/gpt_bundles` as a canonical output directory.
- `tests/test_gpt_bundle.py` updated accordingly.
- `tools/agentic/gpt_bundle.py` succeeds on dirty trees by stashing/restoring (or fails loud with `--no-stash`), and reports dirty/stash status.
- `make test-fast` passes.

## Plan
1. Update the `gpt-bundle` output path in `Makefile` and verify required files still included.
2. Adjust `docs/DOCS_AND_LOGGING_SYSTEM.md` to remove `docs/gpt_bundles` and add the new canonical bundle location.
3. Update `tests/test_gpt_bundle.py` to assert the new bundle path.
4. Harden `tools/agentic/gpt_bundle.py` for dirty-tree bundling with a `--no-stash` option and stash restore checks.
5. Run the provided test command and regenerate the GPT bundle.
6. Update run log + `PROGRESS.md`.

## Notes
- Avoid touching eval logic, overlay math, or experiment outputs.
