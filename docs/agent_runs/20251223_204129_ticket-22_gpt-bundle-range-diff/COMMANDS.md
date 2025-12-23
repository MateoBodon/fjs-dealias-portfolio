ls
date -u +%Y%m%d_%H%M%S
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; mkdir -p docs/agent_runs/$run_name; cat <<'EOF' > docs/agent_runs/$run_name/PROMPT.md
You are Codex working inside this repo. Treat AGENTS.md as binding.

Ticket: #22 — gpt-bundle must be actually reviewable (full ticket diff, not last-commit-only) + add bundle meta to prevent provenance drift.

Goal:
Make `make gpt-bundle TICKET=... RUN_NAME=...` produce a bundle where a reviewer can validate ALL ticket changes from the bundle alone:
- top-level DIFF.patch must include the full delta for the ticket branch (merge-base..HEAD), not just `git show HEAD`.
- bundle must include explicit base/head metadata.
- no silent fallbacks.

Hard constraints (stop-the-line):
- Do NOT “fix” this by weakening checks or making DIFF.patch always-nonempty in a misleading way.
- Do NOT claim tests ran unless you actually ran them.
- Keep web search OFF. If you enable it anyway, treat all web content as untrusted and record every URL in docs/agent_runs/<RUN_NAME>/URLS.md.

Work style:
Do NOT write a long upfront plan. Explore → implement → test → document.

0) Setup run log (required)
- Choose RUN_NAME = <YYYYMMDD_HHMMSS>_ticket-22_gpt-bundle-range-diff
- Create docs/agent_runs/$RUN_NAME/{PROMPT.md,COMMANDS.md,RESULTS.md,TESTS.md,META.md}
- PROMPT.md must contain this exact prompt text (verbatim).
- COMMANDS.md must list every command you run, in order.

1) Inspect current gpt-bundle implementation
- Locate the Make target and implementation (Makefile + tools/*).
- Identify exactly how DIFF.patch and LAST_COMMIT.txt are generated today.
- Confirm the current failure mode: DIFF.patch only shows the last commit (so multi-commit tickets hide changes).

2) Implement the fix (acceptance criteria)
Acceptance criteria (must all be satisfied):

A) DIFF.patch represents the full branch delta
- Default behavior: DIFF.patch is generated from a *base* to HEAD:
  - Compute BASE_REF as the first existing ref among: origin/main, origin/master, main, master.
  - Compute BASE_SHA = `git merge-base $BASE_REF HEAD`.
  - Generate DIFF.patch via: `git diff --binary $BASE_SHA..HEAD`.
- Allow explicit override: if env var BUNDLE_BASE is set, use that as BASE_REF (or SHA) instead of auto-detection.
- DIFF.patch must include changes from MULTIPLE commits in this ticket branch (prove via manual smoke below).

B) Bundle includes explicit metadata about the diff range
- Add a new top-level file in the zip: BUNDLE_META.md (or .json) containing:
  - run_name, ticket id
  - base_ref, base_sha
  - head_sha
  - diff_command used
  - timestamp_utc

C) Fail loud (non-zero exit) on review-breaking states
- If BASE_REF cannot be resolved AND BUNDLE_BASE is not set → fail with a clear error telling the user to set BUNDLE_BASE.
- If DIFF.patch would be empty → fail.
- If required run log files are missing → fail.
- If required top-level files are missing (AGENTS.md, PROGRESS.md, docs/*, project_state/*, LAST_COMMIT.txt) → fail.

D) Docs/contract updates
- Update docs/DOCS_AND_LOGGING_SYSTEM.md §7 to define DIFF.patch as merge-base..HEAD and require BUNDLE_META.* in bundles.
- Update AGENTS.md “Bundle must be reviewable” to explicitly require “full ticket delta, not last commit only”.

E) Provenance hygiene (fix the foot-gun)
- Ensure the ticket’s PROGRESS.md entry and the run log META.md record the correct final HEAD sha (the one matching LAST_COMMIT.txt).
- Add a tiny helper script if needed (optional) but do NOT overbuild; simplest reliable approach is fine.

3) Add minimal test coverage (must run under make test-fast)
- Add/adjust ONE pytest that would have caught ticket-21’s failure:
  - In a temporary git repo created by the test:
    - create branch main with a base commit
    - create feature branch with TWO commits touching different files
    - call your diff generator to produce a range diff (main..HEAD)
    - assert the resulting patch contains evidence of BOTH commits’ changes (e.g., both filenames or both unique markers).
- Also add a test asserting “missing base ref without BUNDLE_BASE fails loud”.

4) Run required tests
- Run: make test-fast
- Record exact summary in docs/agent_runs/$RUN_NAME/TESTS.md.

5) Manual smoke (required, real repo)
- Ensure your ticket branch has at least 2 commits (you will naturally if you commit code then docs).
- Run: make gpt-bundle TICKET=ticket-22 RUN_NAME=$RUN_NAME
- In COMMANDS.md, include:
  - the zip path produced
  - unzip -l <zip>
  - unzip -p <zip> BUNDLE_META.md (or .json)
  - unzip -p <zip> DIFF.patch | head -n 30
  - unzip -p <zip> DIFF.patch | rg -n "<a file changed in commit 1>"
  - unzip -p <zip> DIFF.patch | rg -n "<a file changed in commit 2>"
  (Goal: prove DIFF.patch includes multi-commit changes.)

6) Update repo logs/docs
- Update PROGRESS.md with a new entry for ticket-22 (commands, tests, artifact paths, and the final bundle path).
- Update docs/CODEX_SPRINT_TICKETS.md:
  - mark ticket-21 as FAIL with the reason (last-commit-only diff hid changes)
  - add ticket-22 and mark it IN-PROGRESS (or TODO)

7) Commit discipline
- Work on a feature branch: codex/ticket-22-gpt-bundle-range-diff
- Small logical commits.
- Every commit body must include: Tests: <exact commands run>

8) Finish
- Ensure working tree clean.
- Generate the final bundle AFTER the final commit:
  - make gpt-bundle TICKET=ticket-22 RUN_NAME=$RUN_NAME
- Record the final bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md.
- STOP. Do not merge.
- Finally, generate the reviewer bundle:
  - make gpt-bundle TICKET=ticket-22 RUN_NAME=$RUN_NAME
(Yes, again, to ensure LAST_COMMIT.txt + DIFF.patch match HEAD.)
EOF
: > docs/agent_runs/$run_name/COMMANDS.md
: > docs/agent_runs/$run_name/RESULTS.md
: > docs/agent_runs/$run_name/TESTS.md
: > docs/agent_runs/$run_name/META.md
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; cat <<'EOF' >> docs/agent_runs/$run_name/COMMANDS.md
ls
date -u +%Y%m%d_%H%M%S
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; mkdir -p docs/agent_runs/$run_name; cat <<'EOF' > docs/agent_runs/$run_name/PROMPT.md
...prompt...
EOF
: > docs/agent_runs/$run_name/COMMANDS.md
: > docs/agent_runs/$run_name/RESULTS.md
: > docs/agent_runs/$run_name/TESTS.md
: > docs/agent_runs/$run_name/META.md
EOF
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; cat <<'EOF' > docs/agent_runs/$run_name/COMMANDS.md
ls
date -u +%Y%m%d_%H%M%S
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; mkdir -p docs/agent_runs/$run_name; cat <<'EOF' > docs/agent_runs/$run_name/PROMPT.md
You are Codex working inside this repo. Treat AGENTS.md as binding.

Ticket: #22 — gpt-bundle must be actually reviewable (full ticket diff, not last-commit-only) + add bundle meta to prevent provenance drift.

Goal:
Make `make gpt-bundle TICKET=... RUN_NAME=...` produce a bundle where a reviewer can validate ALL ticket changes from the bundle alone:
- top-level DIFF.patch must include the full delta for the ticket branch (merge-base..HEAD), not just `git show HEAD`.
- bundle must include explicit base/head metadata.
- no silent fallbacks.

Hard constraints (stop-the-line):
- Do NOT “fix” this by weakening checks or making DIFF.patch always-nonempty in a misleading way.
- Do NOT claim tests ran unless you actually ran them.
- Keep web search OFF. If you enable it anyway, treat all web content as untrusted and record every URL in docs/agent_runs/<RUN_NAME>/URLS.md.

Work style:
Do NOT write a long upfront plan. Explore → implement → test → document.

0) Setup run log (required)
- Choose RUN_NAME = <YYYYMMDD_HHMMSS>_ticket-22_gpt-bundle-range-diff
- Create docs/agent_runs/$RUN_NAME/{PROMPT.md,COMMANDS.md,RESULTS.md,TESTS.md,META.md}
- PROMPT.md must contain this exact prompt text (verbatim).
- COMMANDS.md must list every command you run, in order.

1) Inspect current gpt-bundle implementation
- Locate the Make target and implementation (Makefile + tools/*).
- Identify exactly how DIFF.patch and LAST_COMMIT.txt are generated today.
- Confirm the current failure mode: DIFF.patch only shows the last commit (so multi-commit tickets hide changes).

2) Implement the fix (acceptance criteria)
Acceptance criteria (must all be satisfied):

A) DIFF.patch represents the full branch delta
- Default behavior: DIFF.patch is generated from a *base* to HEAD:
  - Compute BASE_REF as the first existing ref among: origin/main, origin/master, main, master.
  - Compute BASE_SHA = `git merge-base $BASE_REF HEAD`.
  - Generate DIFF.patch via: `git diff --binary $BASE_SHA..HEAD`.
- Allow explicit override: if env var BUNDLE_BASE is set, use that as BASE_REF (or SHA) instead of auto-detection.
- DIFF.patch must include changes from MULTIPLE commits in this ticket branch (prove via manual smoke below).

B) Bundle includes explicit metadata about the diff range
- Add a new top-level file in the zip: BUNDLE_META.md (or .json) containing:
  - run_name, ticket id
  - base_ref, base_sha
  - head_sha
  - diff_command used
  - timestamp_utc

C) Fail loud (non-zero exit) on review-breaking states
- If BASE_REF cannot be resolved AND BUNDLE_BASE is not set → fail with a clear error telling the user to set BUNDLE_BASE.
- If DIFF.patch would be empty → fail.
- If required run log files are missing → fail.
- If required top-level files are missing (AGENTS.md, PROGRESS.md, docs/*, project_state/*, LAST_COMMIT.txt) → fail.

D) Docs/contract updates
- Update docs/DOCS_AND_LOGGING_SYSTEM.md §7 to define DIFF.patch as merge-base..HEAD and require BUNDLE_META.* in bundles.
- Update AGENTS.md “Bundle must be reviewable” to explicitly require “full ticket delta, not last commit only”.

E) Provenance hygiene (fix the foot-gun)
- Ensure the ticket’s PROGRESS.md entry and the run log META.md record the correct final HEAD sha (the one matching LAST_COMMIT.txt).
- Add a tiny helper script if needed (optional) but do NOT overbuild; simplest reliable approach is fine.

3) Add minimal test coverage (must run under make test-fast)
- Add/adjust ONE pytest that would have caught ticket-21’s failure:
  - In a temporary git repo created by the test:
    - create branch main with a base commit
    - create feature branch with TWO commits touching different files
    - call your diff generator to produce a range diff (main..HEAD)
    - assert the resulting patch contains evidence of BOTH commits’ changes (e.g., both filenames or both unique markers).
- Also add a test asserting “missing base ref without BUNDLE_BASE fails loud”.

4) Run required tests
- Run: make test-fast
- Record exact summary in docs/agent_runs/$RUN_NAME/TESTS.md.

5) Manual smoke (required, real repo)
- Ensure your ticket branch has at least 2 commits (you will naturally if you commit code then docs).
- Run: make gpt-bundle TICKET=ticket-22 RUN_NAME=$RUN_NAME
- In COMMANDS.md, include:
  - the zip path produced
  - unzip -l <zip>
  - unzip -p <zip> BUNDLE_META.md (or .json)
  - unzip -p <zip> DIFF.patch | head -n 30
  - unzip -p <zip> DIFF.patch | rg -n "<a file changed in commit 1>"
  - unzip -p <zip> DIFF.patch | rg -n "<a file changed in commit 2>"
  (Goal: prove DIFF.patch includes multi-commit changes.)

6) Update repo logs/docs
- Update PROGRESS.md with a new entry for ticket-22 (commands, tests, artifact paths, and the final bundle path).
- Update docs/CODEX_SPRINT_TICKETS.md:
  - mark ticket-21 as FAIL with the reason (last-commit-only diff hid changes)
  - add ticket-22 and mark it IN-PROGRESS (or TODO)

7) Commit discipline
- Work on a feature branch: codex/ticket-22-gpt-bundle-range-diff
- Small logical commits.
- Every commit body must include: Tests: <exact commands run>

8) Finish
- Ensure working tree clean.
- Generate the final bundle AFTER the final commit:
  - make gpt-bundle TICKET=ticket-22 RUN_NAME=$RUN_NAME
- Record the final bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md.
- STOP. Do not merge.
- Finally, generate the reviewer bundle:
  - make gpt-bundle TICKET=ticket-22 RUN_NAME=$RUN_NAME
(Yes, again, to ensure LAST_COMMIT.txt + DIFF.patch match HEAD.)
EOF
: > docs/agent_runs/$run_name/COMMANDS.md
: > docs/agent_runs/$run_name/RESULTS.md
: > docs/agent_runs/$run_name/TESTS.md
: > docs/agent_runs/$run_name/META.md
EOF
python - <<'PY' (rewrite docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
sed -n '1,5p' docs/agent_runs/$run_name/COMMANDS.md
git status -sb
git checkout -b codex/ticket-22-gpt-bundle-range-diff
rg -n "gpt-bundle" -S Makefile tools docs | cat
sed -n '620,740p' Makefile
sed -n '1,240p' tools/gpt_bundle.py
sed -n '120,210p' docs/DOCS_AND_LOGGING_SYSTEM.md
rg -n "Bundle must be reviewable" -n AGENTS.md
sed -n '30,60p' AGENTS.md
sed -n '1,200p' docs/CODEX_SPRINT_TICKETS.md
rg -n "gpt_bundle|gpt-bundle" tests -S
sed -n '1,220p' tests/test_gpt_bundle.py
sed -n '1,220p' tests/test_gpt_bundle_diff.py
apply_patch (tools/gpt_bundle.py range-diff + meta)
apply_patch (Makefile gpt-bundle meta + range diff)
apply_patch (tests/test_gpt_bundle.py add BUNDLE_META)
apply_patch (tests/test_gpt_bundle_diff.py range diff tests)
apply_patch (docs/DOCS_AND_LOGGING_SYSTEM.md bundle range meta)
apply_patch (AGENTS.md bundle full delta)
apply_patch (docs/CODEX_SPRINT_TICKETS.md ticket 21/22 status)
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
apply_patch (tests/test_gpt_bundle_diff.py init fallback)
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git status -sb
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git add Makefile tools/gpt_bundle.py tests/test_gpt_bundle.py tests/test_gpt_bundle_diff.py
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git commit -m "gpt-bundle: range diff + meta output" -m "Tests: not run (not yet)"
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git add AGENTS.md docs/DOCS_AND_LOGGING_SYSTEM.md docs/CODEX_SPRINT_TICKETS.md
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git commit -m "docs: clarify gpt-bundle range diff" -m "Tests: not run (not yet)"
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
make test-fast
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; cat <<'EOF' > docs/agent_runs/$run_name/TESTS.md
- Command: `make test-fast`
- Result: 71 passed, 168 deselected, 1 warning in 13.15s
- Warning: PytestConfigWarning: Unknown config option: timeout
EOF
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git status -sb
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git rev-parse HEAD
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
date -u +%Y-%m-%dT%H:%M:%SZ
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; cat <<'EOF' > docs/agent_runs/$run_name/META.md
run_name: 20251223_204129_ticket-22_gpt-bundle-range-diff
ticket: ticket-22
branch: codex/ticket-22-gpt-bundle-range-diff
git_sha: d4bc504627edee0af8ec7e8dca028dd51f9ed02e
timestamp_utc: 2025-12-23T21:21:10Z
EOF
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git status -sb
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
date -u +%Y-%m-%dT%H:%MZ
python - <<'PY' (prepend PROGRESS.md ticket-22 entry)
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; cat <<'EOF' > docs/agent_runs/$run_name/RESULTS.md
- Updated gpt-bundle to emit merge-base range diffs, include BUNDLE_META.md, and fail loud on missing base refs or empty diffs.
- Bundle (manual smoke): `docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip`.
- Bundle (reviewer): `docs/gpt_bundles/20251223_212700_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip`.
EOF
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git add PROGRESS.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git commit -m "chore: log ticket-22 run" -m "Tests: make test-fast"
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
python - <<'PY' (update PROGRESS.md git sha to 878b2ce1c5a972b2fb9c6dd09cb6e2f5754a77d0)
python - <<'PY' (update META.md git_sha to 878b2ce1c5a972b2fb9c6dd09cb6e2f5754a77d0)
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
BUNDLE_STAMP=20251223_212500 make gpt-bundle TICKET=ticket-22 RUN_NAME=20251223_204129_ticket-22_gpt-bundle-range-diff
unzip -l docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip BUNDLE_META.md
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | head -n 30
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | rg -n "tools/gpt_bundle.py"
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | rg -n "AGENTS.md"
BUNDLE_STAMP=20251223_212700 make gpt-bundle TICKET=ticket-22 RUN_NAME=20251223_204129_ticket-22_gpt-bundle-range-diff
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git status -sb
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git add PROGRESS.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/META.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md
python - <<'PY' (append to docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md)
git add docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md
git commit -m "chore: finalize ticket-22 log" -m "Tests: make test-fast"
git status -sb
git add docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md
git commit -m "chore: log final commands" -m "Tests: make test-fast"
BUNDLE_STAMP=20251223_212500 make gpt-bundle TICKET=ticket-22 RUN_NAME=20251223_204129_ticket-22_gpt-bundle-range-diff
unzip -l docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip BUNDLE_META.md
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | head -n 30
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | rg -n "tools/gpt_bundle.py"
unzip -p docs/gpt_bundles/20251223_212500_ticket-22_20251223_204129_ticket-22_gpt-bundle-range-diff.zip DIFF.patch | rg -n "AGENTS.md"
BUNDLE_STAMP=20251223_212700 make gpt-bundle TICKET=ticket-22 RUN_NAME=20251223_204129_ticket-22_gpt-bundle-range-diff
git rev-parse HEAD
python - <<'PY' (append COMMANDS.md)
python - <<'PY' (update PROGRESS.md sha to 4589320...)
python - <<'PY' (update META.md sha to 4589320...)
python - <<'PY' (append COMMANDS.md)
git add PROGRESS.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/META.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md
python - <<'PY' (append COMMANDS.md)
git commit -m "chore: sync ticket-22 sha" -m "Tests: not run (not yet)"
make test-fast
python - <<'PY' (append COMMANDS.md)
run_name=20251223_204129_ticket-22_gpt-bundle-range-diff; cat <<'EOF' > docs/agent_runs/$run_name/TESTS.md\n- Command: `make test-fast`\n- Result: 71 passed, 168 deselected, 1 warning in 13.03s\n- Warning: PytestConfigWarning: Unknown config option: timeout\nEOF
python - <<'PY' (append COMMANDS.md)
git status -sb
python - <<'PY' (append COMMANDS.md)
python - <<'PY' (update PROGRESS.md bundle stamps + test time)
python - <<'PY' (rewrite RESULTS.md with new bundle paths)
python - <<'PY' (append COMMANDS.md)
git status -sb
python - <<'PY' (append COMMANDS.md)
git add PROGRESS.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/RESULTS.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/TESTS.md docs/agent_runs/20251223_204129_ticket-22_gpt-bundle-range-diff/COMMANDS.md
python - <<'PY' (append COMMANDS.md)
git commit -m "chore: refresh ticket-22 logs" -m "Tests: make test-fast"
