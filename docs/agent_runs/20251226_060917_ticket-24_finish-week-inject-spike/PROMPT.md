You are working in the repo fjs-dealias-portfolio.

Context: The previous run added deterministic window sampling + per-window diagnostics in experiments/eval/inject_spike.py and added gating reason buckets + diagnostics recording in src/fjs/overlay.py. The DoW inject-spike run stayed flat-zero across μ; gating attribution is dominated by pre-gate reasons: tvec_off_component and tvec_compute_error. Multiple week runs were aborted locally due to long dealias_search runtime.

Goal: Finish the inject-spike diagnostic loop in a research-valid way (no fake fixes). Specifically:
1) Run a COMPLETE week inject-spike μ-curve on a fast host (this machine), producing:
   - curve.csv (μ → detection_rate, acceptance_rate, counts),
   - gating_reasons.csv (aggregated reason counts; include the top pre-gate reasons),
   - windows_detail.csv (per-window diagnostics, but keep it bounded),
   - run.json summary with config hash + git SHA + runtime + CPU/thread env.
2) If the week curve is still flat-zero OR dominated by tvec_compute_error/tvec_off_component, create a minimal deterministic reproducer:
   - Select one failing window deterministically (e.g., first failing window in windows_detail),
   - Save its (matrix, group_labels, overlay config, thresholds) to docs/agent_runs/<RUN_NAME>/artifacts/debug_window.npz (or .json/.npz),
   - Add a tiny script under experiments/debug/ (or similar) to load and run detect_spikes and print full diagnostics.
   - Add a unit test that loads that debug window (or a synthetic equivalent) and asserts we do NOT throw tvec_compute_error (i.e., t_vec completes), without changing acceptance logic.
3) Add profiling to identify the hotspot(s) behind dealias_search:
   - minimal cProfile or perf_counter timing around detect_spikes → dealias_search → mp.t_vec.
   - emit a small profile summary text file into docs/agent_runs/<RUN_NAME>/artifacts/profile.txt.
4) Only after reproducing and profiling: implement the smallest correct fix for tvec_compute_error/tvec_off_component.
   - Acceptable fixes: bracketing/robust root-finding fallback, numeric guards, tighter exception reporting, caching where valid.
   - NOT acceptable: disabling gates, always-accept, always-reject, or skipping computations silently.

Process/engineering requirements:
- Create feature branch: codex/ticket-24_finish-week-inject-spike
- Use small, logical commits.
- In EVERY commit body include: "Tests: <commands>".
- Must run: make test-fast
- Must run: python -m pytest tests/experiments/test_inject_spike.py -q
- Must run a REAL-DATA smoke for week inject-spike (not only synthetic). Use the repo’s small derived datasets.
- Use EXEC_MODE=throughput for the long run; if multiprocessing is used, set BLAS threads to 1 to avoid oversubscription and record these env vars in run.json.

Documentation/logging:
- Create run log dir: docs/agent_runs/<RUN_NAME>/ with PROMPT.md, COMMANDS.md, RESULTS.md, TESTS.md, META.md.
- META must include: git_sha, branch, dataset ids/paths, config hash, EXEC_MODE, worker count, thread env (OMP/MKL/OPENBLAS/NUMEXPR), python version.
- Copy the key small artifacts (curve.csv, gating_reasons.csv, profile summary, 1-2 plots) into docs/agent_runs/<RUN_NAME>/artifacts/ so they are reviewable even if reports/ is gitignored.
- Update PROGRESS.md with a crisp entry including the week curve outcome and the dominant gating reasons.
- If results materially change validity/interpretation, update project_state/CURRENT_RESULTS.md and/or project_state/KNOWN_ISSUES.md.

Finish by generating a new bundle:
  make gpt-bundle TICKET=ticket-24 RUN_NAME=<RUN_NAME>
and record the bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md.

Start by inspecting current inject_spike CLI options (--help) and existing make targets (inject-spike / inject-spike-coarse). Then implement/run the above end-to-end.
