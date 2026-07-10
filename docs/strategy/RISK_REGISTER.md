# Risk Register

last_updated: 2026-07-03
updated_by: Codex T-000
source_event: T-000 install AI Project OS v2

| Risk | Severity | Status | Owner | Mitigation | Escalation Trigger |
|---|---|---|---|---|---|
| Stale docs confuse current strategy | High | Active | Pro/Heavy | v2 canonical docs live in `docs/strategy/`; pre-v2 docs are archived and indexed. | A future ticket cites archived docs as current truth without checking v2 docs. |
| T-012 recovered evidence is over-ratified | High | Active | Pro/Heavy | Treat as scientifically useful but not cleanly passed review; require ratification ticket before stronger claims. | Advisor/paper language presents T-012 as cleanly approved. |
| Weekly/oneway detector responsiveness remains flat-zero | High | Active | Pro | Keep injection sensitivity blocker visible in `project_state/CLAIMS_AND_EVIDENCE.md`. | Any plan expands grids before resolving or explaining this blocker. |
| Raw data or bulky generated outputs leak into context bundles | Medium | Active | Codex | Bundle generator excludes raw data and large artifacts, indexing them by path/size instead. | Bundle contains `data/*.csv`, parquet payloads, or large report trees without explicit need. |
| Validation environment drift | Medium | Active | Codex/Heavy | Run and record `make test-fast`, targeted bundle tests, and archive/bundle manifest checks. | Fast test suite cannot run or has missing dependencies. |
| Process drag from over-documentation | Medium | Watch | Pro/Heavy | Single-home docs, concise placeholders, and one rigorous review gate per execution unit. | Future tickets update many docs without a real state change. |
