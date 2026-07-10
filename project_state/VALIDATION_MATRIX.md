# Validation Matrix

last_updated: 2026-07-03
updated_by: Codex T-000
source_event: T-000 install AI Project OS v2

## Validation Levels

| Level | Use | Typical Commands | What It Proves | What It Does Not Prove |
|---|---|---|---|---|
| L0 smoke | Docs/scripts changed only | `git status --short`; JSON/zip manifest checks | Files exist, manifests parse, bundles are readable. | Runtime correctness or research validity. |
| L1 targeted | Tooling or narrow behavior | Targeted pytest files; bundle generator invocation | Changed utility works on expected paths. | Full suite health or scientific claims. |
| L2 fast suite | Normal implementation gate | `. .venv/bin/activate && make test-fast` | Unit-marked test surface passes locally. | Slow/integration runs, expensive experiments, or publishable evidence. |
| L3 integration/reproduction | Runner/output changes | `make run:equity_nested_smoke_tiny`, `EXEC_MODE=deterministic make rc-lite-sanity` | Selected runners produce expected artifacts under deterministic settings. | Full research campaign validity. |
| L4 release/claim audit | Advisor/paper-facing claims | Artifact-specific reproduction, data hash checks, claim/evidence review | A claim is backed by current artifacts and valid comparison rules. | Future robustness or strategic value. |

## Standard Local Commands

| Command | Expected Cost | Use |
|---|---|---|
| `git status --short` | seconds | Confirm current tree state. |
| `python3 -m json.tool <file> >/dev/null` | seconds | Validate JSON manifests. |
| `python3 tools/agentic/ai_os_bundle.py --profile project_state_audit` | seconds | Build Pro-facing state bundle. |
| `python3 tools/agentic/ai_os_bundle.py --profile review --run-log <path>` | seconds | Build Heavy review bundle. |
| `. .venv/bin/activate && pytest -q tests/test_ai_os_bundle.py tests/test_gpt_bundle.py` | seconds/minutes | Targeted bundle-tool regression checks. |
| `. .venv/bin/activate && make test-fast` | minutes | Minimum repo unit gate before merge. |
| `. .venv/bin/activate && make validate-runlogs` | seconds/minutes | Validate agent run-log schema. |
| `. .venv/bin/activate && make check-data-policy` | seconds/minutes | Check data policy guardrails. |

## T-000 Required Checks

| Check | Required | Reason |
|---|---|---|
| `git status --short` | yes | Baseline and final changed-file accounting. |
| Archive manifest JSON parse | yes | Proves archive manifest is machine-readable. |
| Project State Audit Bundle build | yes | Acceptance criterion for T-000. |
| T-000 Review Bundle build | yes | Acceptance criterion for Heavy review. |
| Targeted bundle tests | yes | New tooling regression coverage. |
| `make test-fast` | yes if environment supports it | Existing repo minimum gate. |

## Claim Validation Rules

- Claims about real research effects require uncapped, comparison-valid artifacts and effective sample counts.
- Capped runs can support diagnostics but not headline claims.
- Acceptance/detection-zero runs cannot support treatment-effect claims.
- Recovered T-012 outputs require ratification before being used as clean approved evidence.
