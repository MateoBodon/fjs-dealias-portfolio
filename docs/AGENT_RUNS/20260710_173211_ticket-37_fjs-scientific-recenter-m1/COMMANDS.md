# Commands

Material commands, in execution order:

1. `project-os-v3 status --project-id fjs` and
   `project-os-v3 context --project-id fjs`
   - Resolved the registered canonical SSD repo and matching active scientific
     recenter goal; no competing lifecycle was created.
2. `git worktree add -b portfolio/fjs-recenter-m1-20260710 /Volumes/Storage/Projects/fjs/worktrees/fjs-recenter-m1-20260710 193a325dc681ebc4da67b44715a92e4f63113019`
   - Created the isolated clean writer worktree from the pinned commit.
3. Read `AGENTS.md`, current strategy/state/ticket docs, implementation and
   tests, `CONTROL.md`, the user-goal context, shared WRDS/public manifests, and
   recovered T-012/Ticket 24 artifacts.
4. Hash/readback checks using `shasum -a 256`, `wc`, CSV/JSON inspection, and
   Git object inspection.
5. `python3 -m py_compile src/fjs/detector_contract.py src/fjs/dealias.py src/fjs/overlay.py experiments/eval/run.py experiments/eval/config.py experiments/eval/inject_spike.py experiments/eval/sensitivity.py experiments/daily/run.py`
   - Passed before documentation freeze.
6. `pytest -q tests/fjs/test_detector_contract.py tests/fjs/test_overlay.py`
   - Passed, with the predeclared historical flat-zero promotion test recorded
     as one strict expected failure.
7. `pytest -q tests/experiments/test_eval_run.py -k 'assets_top or ranked_universe or paper_config_path_loads' tests/fjs/test_detector_contract.py tests/fjs/test_overlay.py`
   - First run exposed missing panel provenance in `run.json`; fixed it.
   - Rerun passed (`4 passed`). The `-k` expression applies globally, so this
     is only the targeted ranked-universe subset, not the final combined suite.
8. `ruff check src/fjs/detector_contract.py tests/fjs/test_detector_contract.py`
   and `ruff check --select F,E9 <all changed Python paths>`
   - Passed. The latter also removed four pre-existing F-class findings in
     already-touched files without changing behavior.
9. `pytest -q tests/fjs/test_detector_contract.py tests/fjs/test_overlay.py tests/experiments/test_eval_run.py tests/experiments/test_inject_spike.py tests/experiments/test_daily_grouping.py`
   - One unrelated existing fallback-loader fixture failed because its 14-row
     input is shorter than the unchanged 22-row loader minimum. The same exact
     test and failure were reproduced at pinned base `193a325d`.
10. The same focused command with
    `-k 'not holdout_empty_windows_do_not_trigger_window_coverage_cap'`
    - Passed all 78 collected tests with the one intentional strict XFAIL.
11. `make test-fast`
    - Passed: `97 passed, 178 deselected, 1 xfailed`.
12. `pytest -m 'unit or integration' -q`
    - Passed all 112 collected native tests, including one intentional strict
      XFAIL for the historical flat-zero promotion check.
13. `make validate-runlogs`
    - Passed; historical legacy-META warnings remain non-fatal.
14. `make check-data-policy`
    - The isolated worktree correctly failed because ignored local return CSVs
      are not copied into new worktrees. Running the same tracked validator from
      the canonical SSD repo, where the registered local files live, passed.
15. Final checkpoint, Project OS evidence binding, push, and remote object
    readback are reported at administrator handoff.

No empirical evaluation, CRSP-scale job, or memory-heavy process was launched.
