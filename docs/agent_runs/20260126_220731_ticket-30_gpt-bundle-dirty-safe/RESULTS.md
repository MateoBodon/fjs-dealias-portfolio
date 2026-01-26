# Results
- `tools/agentic/gpt_bundle.py` now stashes/restores dirty trees (with `--no-stash` override) and reports dirty/stash/bundle path; bundle meta records original dirty state.
- `make gpt-bundle` output path is now `artifacts/_local/gpt_bundles/`, and docs/test coverage updated to match.
- Added a unit test asserting stash invocation and artifacts/_local bundle targeting.
- Manual validation (dirty tree) succeeded; status restored with no diff.
- Bundle (manual dirty-tree validation): `artifacts/_local/gpt_bundles/20260126_221517_TICKET-DIRTY-TEST_20260126_050923_ticket-29_gpt-bundle-dirty-check.zip`.
