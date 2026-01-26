# Prompt

User request (2026-01-26):

"lets do this first, then finish that. You are in the repo root. Goal: ensure GPT bundle generation is NEVER blocked by a dirty working tree, while staying compliant with TRACKING_POLICY.md (scratch outputs ignored, curated docs tracked).

Hard rules:
- Do NOT delete any local artifacts/data/logs.
- Do NOT commit unrelated pre-existing changes.
- Bundle outputs must go to scratch (ignored): artifacts/_local/gpt_bundles/
- Bundling must be reproducible and safe:
  - If the repo is dirty, the bundler must temporarily stash and restore, OR bundle from HEAD without touching the working tree.
- Do not use .git/info/exclude for shared rules.

Steps:

1) Baseline diagnostics (include in final report):
   - git status --porcelain
   - ls -la tools/agentic 2>/dev/null || true
   - ls -la scripts 2>/dev/null || true
   - rg -n "gpt_bundle|bundle" -S . || true

2) Identify the bundler entrypoint(s):
   - Prefer tools/agentic/gpt_bundle.py if present.
   - Else scripts/gpt_bundle.py if present.
   - Else search for a python entrypoint that creates bundles/zips for Prompt 3.
   Record the chosen entrypoint path(s).

3) Enforce canonical bundle output location + ignores:
   - Ensure directory exists: artifacts/_local/gpt_bundles/
   - Ensure .gitignore ignores:
     - artifacts/_local/
     - artifacts/_local/gpt_bundles/
   If missing, update .gitignore (policy-compliant). Do not change other ignore policy.

4) Modify the bundler so it never blocks on dirty trees:
   Implement ONE of these approaches (choose the least invasive given current code):

   Option A (recommended): automatic stash wrapper inside bundler
   - Add a helper that runs:
     - save status_before = `git status --porcelain`
     - if dirty:
         - git stash push -u -m "temp: gpt_bundle <ticket or timestamp>"
         - verify clean
     - run bundling logic (zip generation)
     - if stashed:
         - git stash pop
         - verify status_after matches status_before exactly (string compare)
   - If mismatch, abort with clear error and instructions; do NOT drop the stash silently.

   Option B: bundle from HEAD (git archive) + explicit extra files
   - Use git archive HEAD to generate tracked snapshot.
   - Optionally include docs/tickets/<ticket>.md and docs/agent_runs/<run>/ if requested.
   - This avoids any stash interaction.
   - Still write final zip to artifacts/_local/gpt_bundles/.

   Requirements regardless of option:
   - Provide a CLI flag: --allow-dirty (default true) OR --no-stash (to disable stash behavior).
   - Provide clear console output that states:
     - dirty status detected (yes/no)
     - stash used (yes/no)
     - bundle output path
   - Never write bundles under docs/ (tracked). Scratch only.

5) Update docs so future agents don’t regress:
   - In docs/agent_runs/README.md (or AGENTS.md if that’s where protocol lives), add a short note:
     - Bundles are emitted to artifacts/_local/gpt_bundles/ and are ignored by design.
     - Bundling is allowed even with dirty trees; the tool will stash temporarily or bundle from HEAD.

6) Add a small self-check / test (keep it lightweight):
   - If repo uses pytest: add a fast unit test that mocks dirty detection and ensures:
     - stash wrapper is invoked (or archive path chosen)
     - output zip path is under artifacts/_local/gpt_bundles/
   - If no test framework, add a script-level “--self-check” mode that:
     - prints chosen output path and dirty-tree strategy
     - exits 0

7) Validate manually (do not modify unrelated files):
   - Create a tiny temporary untracked file: tmp/_bundle_dirty_test.txt (ensure tmp/ is ignored per policy)
   - Run the bundler with a dummy ticket id:
     python3 <bundler_path> --zip --ticket TICKET-DIRTY-TEST
   - Confirm:
     - It succeeds even though repo is dirty (because tmp file exists)
     - It produces a zip under artifacts/_local/gpt_bundles/
     - After it finishes, git status --porcelain is unchanged from before the run.

8) Stage ONLY relevant changes:
   - bundler code changes
   - .gitignore (only if you had to add the artifacts/_local/gpt_bundles ignore)
   - docs note change
   - any new lightweight test/self-check file
   Do NOT stage unrelated pre-existing changes.

9) Commit using repo’s required commit message style (check AGENTS.md / existing conventions):
   - Title should be something like: "ticket-XX: make gpt_bundle dirty-tree safe"
   - Body must include:
     - Tests: <what you ran>
     - Artifacts: n/a (bundles are ignored scratch)

Deliverables:
- Bundler no longer blocks on dirty working tree
- Bundles always land in artifacts/_local/gpt_bundles/
- Documentation updated so future agents follow this
- Proof in final message: before/after git status unchanged + example output zip path"
