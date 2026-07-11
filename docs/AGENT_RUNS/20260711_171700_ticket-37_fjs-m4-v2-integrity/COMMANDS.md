# Commands

Material commands and outcomes, in execution order:

1. `project-os-v3 status --project-id fjs --summary` and the context resolver
   - Recovered active goal `goal_71e41ee4ddc6` and canonical root
     `/Volumes/Storage/Projects/fjs/repo`.
   - Confirmed sibling worktree execution must continue writing Project OS
     events through the canonical root.
2. `project-os-v3 route-plan --project-id fjs --goal-id goal_71e41ee4ddc6`
   - Returned `trusted routing configuration has no exact project/goal task
     envelope`; no admission was created or consumed.
3. Git/worktree and v2 source inspection, followed by focused tests
   - Existing five mocked checkpoint tests passed.
   - A real smoke exposed a reversed exact-binomial upper-bound bisection and
     attribution against a newly sampled, unrelated direction.
4. Deterministic implementation repair
   - Corrected the Clopper-Pearson upper bound.
   - Returned the actual planted and nuisance directions from the simulator.
   - Paired trial seeds across signal strengths and bound separate null/alt
     counts.
   - Bound checkpoints to hashes of the exact executable bytes in addition to
     Git `HEAD^{tree}`.
   - Repaired process-worker argument construction and terminate/wait cleanup.
   - Replaced smoke-derived AWS dollar extrapolation with an explicit
     not-estimable state requiring a fresh price and stratified benchmark.
   - Added fail-closed full-execution readiness blockers.
5. Manifest generation and byte replay
   - Generated `fjs_m4_full_target_between_v2.json` and
     `fjs_m4_smoke_target_between_v2.json`.
   - Independent regeneration plus `cmp` reproduced both byte-for-byte.
6. Real two-process interrupt/resume/fresh smoke under
   `/tmp/fjs-m4-v2-commit.SWsXTD`
   - Intentional interrupt returned `1`.
   - Resume and fresh stable reducers were exactly equal at SHA-256
     `049408720a08c3178a4ef0f161998235149cd5ee9f6dbcfdefbfdfe1a11821a6`.
   - No worker process survived the intentional interruption or completion.
7. Validation
   - Focused M4/synthetic suite passed.
   - `make detector-reference-gate` passed with `issue_count=0`.
   - `make test-fast` passed: `117 passed, 187 deselected`.
   - Native `pytest -m 'unit or integration' -q` passed with return code `0`.
   - Changed-path Ruff, compile, `git diff --check`, and canonical
     `make check-data-policy` passed.
8. `git commit -m "fix: make FJS M4 calibration prep fail closed"`
   - Created code/manifest commit
     `2bc8a61e6a3737082f0849f83ead6bacc6704997`, tree
     `c4b81988284201110b3f785553d0aec46737dacd`.

No AWS, WRDS, broad grid, external publication, or push occurred.
