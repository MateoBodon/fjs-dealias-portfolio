# Prompt

## Verdict

FAIL

## Required Fixes

1. Fix ticket-35 runlog metadata so `docs/agent_runs/20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance/META.json` and `META.md` set `git_sha_after` to the true post-commit SHA `71a700bb15a7f39b70a705215d5258e2d24549f3`.
2. Add a guardrail so `make gpt-bundle` or `make validate-runlogs` fails loud for timestamped runs (`>= 20260216_000000`) when `META.json.git_sha_after` is missing/TBD or does not match bundle `head_sha` for the run being bundled.
3. Append a corrective `PROGRESS.md` errata entry that marks `artifacts/_local/gpt_bundles/20260217_011000_35_20260217_010137_ticket-35_fix-ticket34-canonical-bundle-provenance.zip` as superseded and records the new canonical bundle path + head SHA.
4. Re-run and record:
   - `. .venv/bin/activate && make validate-runlogs`
   - `. .venv/bin/activate && make test-fast`
5. Rebuild a new bundle for Ticket-36 with this run.

## Ticket-36

Title: Enforce runlog META.json SHA correctness + fix ticket-35 meta drift
Goal: Make run logs audit-truthful by fixing ticket-35 incorrect SHAs and enforcing guardrails so future bundles cannot ship with mismatched runlog metadata.
