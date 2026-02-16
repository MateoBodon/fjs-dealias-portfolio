# Tests

- `rg -n "BLOCKED: the exact \\Analysis\\.md source text" docs/gpt_outputs/20260216_project_review_full.md`
  - Result: pass (no matches).
- `git check-ignore -v docs/gpt_outputs/20260216_project_review_full.md`
  - Result: pass (no output, file not ignored).
- `python3 -c 'import json; print(json.load(open("docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/META.json"))["git_sha_after"])'`
  - Result: pass (`7003d53fc31cf00e1a7b2032a620abd0e39a7d53`).
- `. .venv/bin/activate && make validate-runlogs`
  - Result: pass.
- `. .venv/bin/activate && make test-fast`
  - Result: pass (`83 passed, 171 deselected in 22.69s`).
