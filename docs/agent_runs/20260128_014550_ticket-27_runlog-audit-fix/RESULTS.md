# Results

## Summary
- Corrected ticket-27 cleanup run log metadata (`git_sha_after` and `dirty_at_end`) to match bundle evidence.
- Added missing `META.json` for the ticket-27 cleanup run.
- Recorded clean-tree evidence in the ticket-27 cleanup run log notes/tests using bundle metadata.
- Backfilled the missing follow-up run log folder with explicit N/A status and a pointer to this audit run.

## Key outputs
- Path: docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/META.md
- Path: docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/META.json
- Path: docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/RESULTS.md
- Path: docs/agent_runs/20260127_053650_ticket-27_repo-hygiene-cleanup/TESTS.md
- Path: docs/agent_runs/20260127_222035_ticket-27_repo-hygiene-cleanup-followup/
- Bundle: artifacts/_local/gpt_bundles/20260128_015625_ticket-27_20260128_014550_ticket-27_runlog-audit-fix.zip

## Notes
- Bundle evidence source: artifacts/_local/gpt_bundles/20260127_162507_27_20260127_053650_ticket-27_repo-hygiene-cleanup.zip (BUNDLE_META.md).
- Audit-fix bundle: artifacts/_local/gpt_bundles/20260128_015625_ticket-27_20260128_014550_ticket-27_runlog-audit-fix.zip.
- Tests were executed before the metadata edits to avoid pytest cleaning untracked files.
