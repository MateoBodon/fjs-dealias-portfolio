RUN_NAME: 20251222_183526_ticket-07_advisor-ready-dow
branch: feat/ticket-07-advisor-ready-dow
git_sha: 2cb5bfdce66324fff011d994d552a4b9bc42740c (short: 2cb5bfd)
working_tree_clean_start: yes (git status -sb on main was clean)
exec_mode: deterministic
python: 3.12.3

Command (eval run)
- PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out reports/rc-ticket-07-20251222_183800/dow-paper-v1 --exec-mode deterministic

Resolved config
- path: reports/rc-ticket-07-20251222_183800/dow-paper-v1/resolved_config.json
- sha256: 1cbee2de6cb4e98653e6d0be97b85662bc8b2e47d4f7355a5042ac122442d694

Data provenance
- returns registry key: data/returns_daily.csv
  - sha256: 96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197
  - rows: 892529; date range: 2010-01-05 → 2024-12-31
- factors registry key: data/factors/ff5mom_daily.csv
  - sha256: 469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca
  - date range: 2005-01-03 → 2025-08-29
- verification command: python3 scripts/check_data_policy.py

Environment notes
- workspace: /root/fjs-dealias-portfolio
- run outputs: reports/rc-ticket-07-20251222_183800/dow-paper-v1/
