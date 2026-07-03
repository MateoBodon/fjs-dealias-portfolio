# T-012 Recovery Status

Recovered on 2026-07-02 from `20260401_055651_T-012_gpt_bundle.zip`.

## Source

- source bundle: `20260401_055651_T-012_gpt_bundle.zip`
- source bundle sha256: `f2ca8e4a9621d0f72f298a3255e5eefca472638c8c62b06ce16800b9f04f1aad`
- source branch in bundle metadata: `chore/fjs-repo-os-bootstrap-20260324`
- source bundle git sha: `eb6cf4fd3fe65773bc21a1cfd73e8c7d64851f0f`
- generated head recorded in bundle: `97ac10bb009e5df371dd89b6299d935d59cb0816`
- current GitHub recovery base: `f73d8acecfcdd19917d0d0d9e25911cfcf02b54d`

## Recovered Into Git

- T-012 ticket: `docs/tickets/T-012-execute-the-next-daily-dow-empirical-campaign.md`
- T-012 run log, migrated to current convention: `docs/agent_runs/20260331_230430_t-012/`
- T-012 derived configs:
  - `experiments/eval/config.paper_v1_dow_noprewhiten.yaml`
  - `experiments/eval/config.paper_v1_dow_window252.yaml`
  - `experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`
- curated T-012 summary pack: `docs/artifacts/rc-t-012/summary/`
- T-010 advisor memo referenced by T-012: `docs/artifacts/rc-t-010/summary/advisor_decision.md`

## Local Only

The full heavy artifact tree is preserved outside Git at:

```text
/Volumes/Storage/Projects/fjs/_recovery/recovered_artifacts/rc-t-012
```

Large detail surfaces such as `overlay_forensics.csv`, per-run detail CSVs, and PNGs remain local-only.

## Scientific Status

- T-012 completed the four-leg daily DoW matrix under the empirical-only lane.
- Both `w126` control legs reproduce the ratified T-010 full-regime truth.
- Both `w252` legs remain uncapped, full-coverage, comparison-valid, and QLIKE-improving versus baseline for both `ew` and `mv`, with smaller deltas than `w126`.
- The campaign decision is `empirical-lane-still-worth-scaling`.

## Claim Boundary

- Daily DoW remains empirical-only.
- Daily DoW is not detector validation.
- Daily DoW does not restore the original weekly oneway / clean FJS theory story.
- The T-012 review failed because monitoring/audit-trail preservation was not fully review-backed, not because the scientific outputs were shown corrupt.

## Recommended Next Ticket

T-013 should ratify the existing T-012 artifact pack, add reusable long-run monitoring discipline, and lock the next scale-up matrix without rerunning T-012 unless corruption is first shown.
