# Commands

Material commands, in execution order:

1. `project-os-v3 status --project-id fjs` and the Project OS context resolver
   - Reused active goal `goal_c85213aa76be` and the shared canonical writer; no
     competing goal or lifecycle was created.
2. `git status --short`, `git log`, and branch/worktree inspection
   - Confirmed clean starting HEAD `e5399c4e94148abdad2a6585df9e548db9100025`
     on `portfolio/fjs-recenter-m1-20260710`.
3. Exact historical source and artifact review using `git show`, `shasum -a
   256`, CSV/JSON inspection, and the recorded Ticket 24 command log
   - Bound the recovered artifact to source commit `31c05a57...` and determined
     that it used observation-level iid outer-product injection with no
     component-mode provenance.
4. Implemented `src/fjs/reference_oracle.py`,
   `tools/check_fjs_reference.py`, detector treatment-provenance checks, strict
   expected-failure tests, state, and strategy evidence.
5. `black ...`, `ruff check ...`, `python3 -m py_compile ...`, and
   `git diff --check`
   - Passed on every changed Python path and the full patch.
6. `PYTHONPATH=src:. pytest -q tests/fjs/test_reference_oracle.py
   tests/fjs/test_detector_contract.py`
   - Passed with three intentional strict expected failures.
7. `PYTHONPATH=src:. pytest -q tests/fjs/test_reference_oracle.py
   tests/fjs/test_detector_contract.py tests/test_mp.py
   tests/test_mp_edge_and_root.py tests/test_theta_solver.py
   tests/test_balanced.py tests/test_dealias.py tests/fjs/test_overlay.py`
   - Passed the focused reference/production interaction surface with the same
     three intentional strict expected failures.
8. `make test-fast`
   - Passed: `106 passed, 178 deselected, 3 xfailed`.
9. `make detector-reference-gate`
   - Expected non-zero stop: five named production/provenance issues. No gate
     threshold or reference value was weakened.
10. `make validate-runlogs`
    - Passed before this run log was added; final validation is recorded in
      `TESTS.md`.
11. `make check-data-policy` in `/Volumes/Storage/Projects/fjs/repo`
    - Passed against the registered local data authorities. No restricted data
      was copied into this isolated worktree or Git.
12. `git commit -m "feat: add deterministic FJS reference gate"`
    - Created implementation checkpoint
      `ce147d91305155e5d3d7c178465d8e63713ce343` with tree
      `755ed46510939929924ec7fc11871236ddc96082`.
13. Final run-log validation and the evidence commit are recorded in
    `RESULTS.md` and `TESTS.md`. Scoped push, remote fetch, and commit/tree
    readback are reported directly to the Portfolio Administrator.

No empirical evaluation, CRSP-scale job, or memory-heavy process was launched.
