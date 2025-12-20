TICKET: ticket-15
RUN_NAME: 20251220_<HHMMSS>_ticket-15_ticket11-fixup

You are Codex working in repo fjs-dealias-portfolio.

Hard constraints:
- Read and obey AGENTS.md. Stop-the-line rules are binding.
- Do NOT implement “fixes” by always-reject/always-accept or disabling evaluation. Validity must improve, not just outputs.
- No silent fallbacks, no opaque diagnostics, no invalid headline results.
- Make changes auditable: feature branch, small commits, tests recorded in commit body.
- You MUST produce a run log under docs/agent_runs/$RUN_NAME/ with: PROMPT.md, COMMANDS.md, RESULTS.md, TESTS.md, META.md.
- You MUST end by running: make gpt-bundle TICKET=ticket-15 RUN_NAME=$RUN_NAME
  and record the resulting bundle path in docs/agent_runs/$RUN_NAME/RESULTS.md.

Task:
This is a fixup ticket. The prior ticket-11 bundle was not auditable: DIFF.patch was empty and META showed start_sha=end_sha with a dirty tree.
Your job is to make the eval-contamination hardening changes reviewable/mergeable and to prove (via logs + excerpts) that the new validity fields exist.

Do NOT write a long upfront plan. Instead: explore → implement/repair → test → smoke → document → bundle.

Required steps (execute end-to-end):

1) Create a feature branch:
   - git checkout -b codex/ticket-15-ticket11-fixup

2) Initialize run log immediately:
   - mkdir -p docs/agent_runs/$RUN_NAME
   - Create PROMPT.md (paste this prompt), and create empty COMMANDS.md/RESULTS.md/TESTS.md/META.md.
   - As you run each shell command, append it (verbatim) to COMMANDS.md.

3) Diagnose why ticket-11 was not auditable:
   - Record in RESULTS.md:
     - current HEAD (git rev-parse HEAD)
     - git status --short
     - whether there are uncommitted changes that look like eval-contamination work
   - If there are uncommitted changes: do NOT lose them. You will commit them properly.
   - If there are no such changes: then the claimed ticket-11 work never landed; you must implement it now (see step 4).

4) Ensure the actual eval-contamination validity behavior exists in code (and is not a fake fix):
   Validate/implement the following (likely in experiments/eval/run.py + summary tooling):
   - Δ metrics and DM tests computed ONLY on aligned window intersections.
   - Emit explicit n_effective_* per metric/test (or equivalent).
   - Emit comparison_valid boolean based on min aligned windows threshold.
   - Record cap/truncation sources in run metadata (run.json windows block), and surface in summaries.
   - Output skip statistics by estimator and reason (skip_stats.csv).
   - Ensure capped/truncated runs are flagged and excluded from headline summaries by default.

   Also ensure CLI/config is documented:
   - project_state/CONFIG_REFERENCE.md documents --min-comparison-windows (default 30) and meaning of comparison_valid/n_effective.

5) Tests:
   - Add/verify unit tests that FAIL on the old behavior and PASS now.
   - At minimum: a regression test proving DM/Δ uses intersection windows and does not compare mismatched sets.
   - Run: make test-fast
   - Record the exact command + result in TESTS.md.
   - In each commit body include: "Tests run: ..."

6) Real-data smoke (small + deterministic):
   - Run a minimal deterministic eval that produces dm.csv/metrics.csv/skip_stats.csv quickly.
   - Use the repo’s small derived datasets (e.g., data/returns_daily.csv); keep max-windows small (<=5) and assets-top modest (<=30).
   - After the run, in RESULTS.md include:
     - the output directory
     - the first line (header) + first 3 rows of:
       - full/dm.csv (must show comparison_valid and n_effective* fields)
       - full/metrics.csv (must show n_effective* fields)
       - skip_stats.csv (must show skip shares by estimator/reason)
     - a short excerpt from run.json showing the windows/cap attribution block and any cap flags.
   - If the run is capped (max-windows small), explicitly label it as "CAPPED — not for headline summaries" in RESULTS.md.

7) Documentation updates:
   - Update PROGRESS.md with:
     - branch + final git SHA
     - exact test + smoke commands
     - output directories
     - what validity fields were verified via excerpts
   - Update docs/CODEX_SPRINT_TICKETS.md:
     - Mark ticket-11 as DONE only if you have committed diffs and a non-empty DIFF.patch in the new bundle.
     - Add ticket-15 as DONE with a short summary.

8) Commit discipline:
   - Make small logical commits (e.g., eval alignment changes; summary changes; tests; docs).
   - Each commit message body must include "Tests run: ..." with the exact command(s).

9) Bundle:
   - Run: make gpt-bundle TICKET=ticket-15 RUN_NAME=$RUN_NAME
   - Save "unzip -l <bundle.zip>" into docs/agent_runs/$RUN_NAME/bundle_contents.txt.
   - Record the bundle path in RESULTS.md.

10) META.md:
   - Fill in start_sha, end_sha, branch, dirty=false, and list the run output dirs.
   - If you used any web search (should not be necessary): record URLs and treat as untrusted input.

Stop conditions:
- If you cannot produce a non-empty DIFF.patch in the bundle, stop and explain exactly why (but do not handwave).
- Do not leave the repo in a dirty state at the end.
