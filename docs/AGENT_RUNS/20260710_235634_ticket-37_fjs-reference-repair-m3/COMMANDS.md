# Commands

Material commands and checkpoints, in execution order:

1. `project-os-v3 status --project-id fjs` plus the context resolver and a read
   of `/Volumes/Storage/Projects/PROJECT_OS_CONTROL/CONTROL.md`
   - Confirmed the sole writer and began bounded goal `goal_66cfb5598280`.
2. `git status`, branch/worktree inspection, and source/reference review
   - Started clean at `e6df52575a1db65c85372ddaa545ae62698ee507`.
   - Verified that `src/fjs/reference_oracle.py` was unchanged throughout.
3. `git commit -m "docs: predeclare FJS between calibration fixture"`
   - Froze the seed, dimensions, detector settings, two cells, trial count, and
     no-change-on-failure rule at `82d1ffc0b2fc7c4c39e820b7aae3c4ad0bcdb43c`.
4. Repaired `src/fjs/mp.py`, `src/fjs/dealias.py`,
   `src/fjs/theta_solver.py`, reconstruction, overlay, and focused tests.
   - Production checkpoint: `4437571acf4b42bd1f4c7db8a9616b623c5a3a7b`.
   - Generator checkpoint: `d9b522da22206dbd9557c7fd0706bc3adffda0e8`.
   - Bounded/prioritized theta seed checkpoints: `9445a37b2b2a1c20c624a5dab9a5663320f940ca`
     and `9afbb72cb02172080c52ba206ddd73ed2110dedf`.
5. `PYTHONPATH=src:. python3 tools/generate_fjs_between_fixture.py`
   - Executed the frozen 12-pair mechanism fixture against source commit
     `9afbb72cb02172080c52ba206ddd73ed2110dedf`, tree
     `75ce510b7193eee457f825fa57d6417a2c361170`.
6. `PYTHONPATH=src:. python3 tools/generate_fjs_between_fixture.py --check`
   - PASS; output bytes reproduced without mutation.
7. `make detector-reference-gate`
   - PASS with `issue_count=0` using the unchanged oracle and reducer logic.
8. Focused reference/MP/de-alias/theta/reconstruction/overlay tests, then
   `make test-fast` and `pytest -m "unit or integration" -q`
   - PASS; fast suite reported `117 passed, 178 deselected`.
9. `ruff check` over changed Python paths, `git diff --check`,
   `make validate-runlogs`, and canonical `make check-data-policy`
   - PASS; no restricted data was copied to Git or the isolated worktree.
10. Five independent-process timing calls on the same fixed planted panel at
    `4437571`, `9445a37`, and `9afbb72`
    - Median detector times were `5.318868 s`, `0.380708 s`, and `0.395707 s`.
    - The bounded and strongest-first versions were respectively `13.971x`
      and `13.441x` faster than the repaired unbounded checkpoint. The latter
      was `3.940%` slower than the bounded-only version on this microbenchmark;
      its value is deterministic strongest-root selection and robust fixture
      reachability, not a further raw-speed claim.
11. `git commit -m "feat: pass FJS deterministic reference gate"`
    - Created substantive result commit
      `143b74972392a0fa1ba0ece6384a33d2c1a663fd`, tree
      `8315f975e24ef5c387c5b58350ce578097eb9098`.

No broad calibration, real-data experiment, memory-heavy process, or AWS job
was launched.
