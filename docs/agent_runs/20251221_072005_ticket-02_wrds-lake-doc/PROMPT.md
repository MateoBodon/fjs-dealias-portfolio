document what is available in case we ever need it for refrenshing etc, If you want me to document the WRDS lake structure (as you listed) in a repo doc so future GPT runs see it, then go aheard and make sure this entire checklist is complete, once it is merge into main, all docs/info files should not be commited but preserved on local docs/agent_runs/<RUN_NAME>/ exists and contains PROMPT/COMMANDS/RESULTS/TESTS/META (+ DIFF.patch recommended).

 Branch is feat/ticket-02-stop-eval-contamination, commits are small, and each commit body includes Tests run:.

 make test-fast passes (and is recorded in both TESTS.md and commit bodies).

 In a capped smoke run (--max-windows), tools/make_summary.py:

 excludes capped runs from summary_perf.csv and summary_detection.csv

 writes summary/limitations.md with a clear “excluded” section listing paths + cap_sources

 Any run with mv_skip_on_missing_solver is labeled smoke-only in limitations.md (and excluded from headline tables if that’s the implemented policy).

 run.json consistently includes cap_active / cap_sources (explicit false/empty when uncapped), no schema ambiguity.

 Bundle generated and path recorded in docs/agent_runs/<RUN_NAME>/RESULTS.md:

docs/gpt_bundles/<stamp>_ticket-02_<RUN_NAME>.zip

 No “fix” achieved by disabling functionality (no always-exclude / always-accept / turning caps off silently).
