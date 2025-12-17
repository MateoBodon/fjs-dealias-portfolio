# Project State Index

Last refreshed: 2025-12-17 (Codex sweep)

This directory is the knowledge spine of the repo. Each file below is a stable entry point for newcomers and agents; update them incrementally rather than rewriting from scratch.

- `ARCHITECTURE.md` — High-level purpose, directory map, major components, and data/control flow.
- `MODULE_SUMMARIES.md` — Per-module descriptions with key classes/functions and algorithm notes.
- `FUNCTION_INDEX.md` — Table of important public functions/classes with signatures, returns, and dependencies.
- `DEPENDENCY_GRAPH.md` — Textual import graph; highlights high-fan-out modules and coupling risks.
- `PIPELINE_FLOW.md` — End-to-end pipelines (rc/rc-lite/rc-lite-sanity, synthetic, ablations, reporting) with entry commands and outputs.
- `DATAFLOW.md` — Data locations (raw/processed/synthetic), expected formats, caches, and path/env assumptions.
- `EXPERIMENTS.md` — Catalog of experiment types, configs, metrics, and where outputs land.
- `CURRENT_RESULTS.md` — Snapshot of latest metrics/runs, with gaps/recency flags.
- `RESEARCH_NOTES.md` — Theory ↔ implementation bridge, notation map, and conceptual caveats.
- `OPEN_QUESTIONS.md` — Unresolved design/analysis questions to investigate next.
- `KNOWN_ISSUES.md` — Bugs/limitations/perf and stability concerns.
- `ROADMAP.md` — Short/medium/long-term priorities and sequencing dependencies.
- `CONFIG_REFERENCE.md` — Key config files/flags/env vars with meanings and defaults.
- `SERVER_ENVIRONMENT.md` — Runtime expectations (Python/dep stack, hardware, thread caps, data mounts).
- `TEST_COVERAGE.md` — Test suite scope, markers, notable gaps, and how to run.
- `STYLE_GUIDE.md` — Observed/recommended coding + documentation conventions.
- `CHANGELOG.md` — Dated change log for code/doc/experiment structure updates.
