# Commands

1. `sed -n '1,220p' AGENTS.md`
2. `sed -n '1,220p' /home/codex/.codex/skills/runlog-init/SKILL.md`
3. `sed -n '1,220p' /home/codex/.codex/skills/gpt-bundle/SKILL.md`
4. `sed -n '1,260p' PROJECT.md`
5. `sed -n '1,260p' docs/plan/PLAN_OF_RECORD.md`
6. `sed -n '1,260p' docs/plan/NOW.md`
7. `rg --files docs/tickets | rg 'T-012|012|ticket-12|ticket-012'`
8. `sed -n '1,320p' docs/tickets/T-012-execute-the-next-daily-dow-empirical-campaign.md`
9. `sed -n '1,240p' docs/recenter/RECENTER_MEMO.md`
10. `sed -n '1,240p' docs/recenter/OPEN_QUESTIONS.md`
11. `sed -n '1,240p' project_state/ARCHITECTURE.md`
12. `sed -n '1,240p' project_state/RUNBOOK.md`
13. `git status --short`
14. `test -d reports/rc-t-012 && echo EXISTS || echo MISSING`
15. `rg -n "T-012|t-012|ticket-012|ticket-12" reports/_runs -S`
16. `ls -1 reports/_runs | tail -n 20`
17. `sed -n '1,220p' experiments/eval/config.paper_v1.yaml`
18. `sed -n '1,220p' experiments/eval/config.paper_v1_dow_noprewhiten.yaml`
19. `python3 tools/agentic/runlog_init.py --ticket "T-012" --summary "Execute the next daily dow empirical campaign"`
20. `sed -n '1,220p' reports/_runs/20260331_220201_t-011/COMMANDS.md`
21. `sed -n '1,260p' reports/_runs/20260331_220201_t-011/RESULTS.md`
22. `sed -n '1,220p' reports/_runs/20260331_220201_t-011/TESTS.md`
23. `sed -n '1,260p' tools/make_summary.py`
24. `make -f Makefile.agentic init-runlog TICKET=T-012`
25. `sed -n '1,260p' project_state/CURRENT_STATE.md`
26. `sed -n '1,260p' project_state/KNOWN_ISSUES.md`
27. `sed -n '1,260p' PROGRESS.md`
28. `sed -n '1,260p' reports/rc-t-010/summary/advisor_decision.md`
29. `ls -1 reports/_runs/20260331_230430_t-012 && sed -n '1,240p' reports/_runs/20260331_230430_t-012/META.json && sed -n '1,240p' reports/_runs/20260331_230430_t-012/PROMPT.md && sed -n '1,240p' reports/_runs/20260331_230430_t-012/COMMANDS.md && sed -n '1,240p' reports/_runs/20260331_230430_t-012/RESULTS.md && sed -n '1,240p' reports/_runs/20260331_230430_t-012/TESTS.md`
30. `sed -n '1,260p' reports/rc-t-010/summary/t010_full_regime_comparison.csv`
31. `rg -n "t010_full_regime_comparison|advisor_decision|campaign_decision|t012_full_regime_comparison|summary_detection|overlay_forensics" -S .`
32. `sed -n '1,320p' scripts/make_gpt_bundle.py`
33. `sed -n '1,260p' reports/_runs/20260331_014033_t-010/COMMANDS.md`
34. `sed -n '1,260p' reports/_runs/20260331_014033_t-010/RESULTS.md`
35. `sed -n '1,260p' reports/_runs/20260331_014033_t-010/TESTS.md`
36. `rg -n "headline_gate_pass|qlike_improves_vs_baseline|mandatory_baselines_present|full_regime_comparison" reports/_runs docs/tickets tests tools -S`
37. `ls -1 experiments/eval | sort`
38. `git ls-files --error-unmatch experiments/eval/config.paper_v1_dow_noprewhiten.yaml && (git ls-files --error-unmatch experiments/eval/config.paper_v1_dow_window252.yaml || true) && (git ls-files --error-unmatch experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml || true)`
39. `sed -n '1,260p' docs/tickets/T-010-run-the-matched-daily-dow-robustness-pair-and-write-the-advisor-go-no-go-memo.md`
40. `sed -n '1,220p' reports/rc-t-010/dow-paper-v1_ff5mom/resolved_config.json`
41. `sed -n '1,220p' reports/rc-t-010/dow-paper-v1_noprewhiten/resolved_config.json`
42. `python - <<'PY' ... yaml diff assertion for base/no-prewhiten/window252/window252_noprewhiten configs ... PY`
43. `bash -lc '. .venv/bin/activate && make test-fast'`
44. `uptime`
45. `free -h`
46. `nproc`
47. `ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | head -n 20`
48. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
49. `test ! -e reports/rc-t-012/dow-paper-v1_ff5mom_w126 && echo ABSENT`
50. `test ! -e reports/rc-t-012/dow-paper-v1_noprewhiten_w126 && echo ABSENT`
51. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
52. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
53. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_ff5mom_w126 --exec-mode deterministic'`
54. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_noprewhiten.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w126 --exec-mode deterministic'`
55. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
56. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
57. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
58. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
59. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
60. `tail -n 20 reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.log`
61. `tail -n 20 reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.log`
62. `sed -n '1,40p' reports/rc-t-010/summary/summary_perf.csv`
63. `cat reports/rc-t-010/summary/completeness.json`
64. `cat reports/rc-t-010/summary/kill_criteria.json`
65. `sed -n '1,40p' reports/rc-t-010/dow-paper-v1_ff5mom/full/metrics.csv`
66. `sed -n '1,40p' reports/rc-t-010/dow-paper-v1_noprewhiten/full/metrics.csv`
67. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.json`
68. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.json`
69. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
70. `git add -- experiments/eval/config.paper_v1_dow_window252.yaml experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml`
71. `git status --short -- experiments/eval/config.paper_v1_dow_window252.yaml experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml reports/_runs/20260331_230430_t-012`
72. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
73. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
74. `tail -n 20 reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.log`
75. `tail -n 20 reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.log`
76. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
77. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
78. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
79. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
80. `tail -n 20 reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.log`
81. `tail -n 20 reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.log`
82. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
83. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
84. `sed -n '1,80p' reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.json`
85. `sed -n '1,80p' reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.json`
86. `rg -n "status.*initialized|validate-runlog|run log" tests/test_validate_runlog.py tools -S`
87. `sed -n '1,260p' tests/test_validate_runlog.py`
88. `make -f Makefile.agentic help`
89. `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
90. `make -f Makefile.agentic gpt-bundle TICKET=T-012 RUN_DIR=reports/_runs/20260331_230430_t-012`
91. `python - <<'PY' ... interim bundle assertion over reports/_bundles/20260331_234217_T-012_gpt_bundle.zip ... PY`
92. `make -f Makefile.agentic gpt-bundle TICKET=T-012 RUN_DIR=reports/_runs/20260331_230430_t-012`
93. `git add -- scripts/make_gpt_bundle.py`
94. `python - <<'PY' ... interim bundle assertion over reports/_bundles/20260331_234249_T-012_gpt_bundle.zip ... PY`
95. `bash -lc '. .venv/bin/activate && pytest -q tests/test_gpt_bundle.py'`
96. `date -u '+%Y%m%d_%H%M%S'`
97. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
98. `ps -eo pid,etime,pcpu,pmem,cmd | rg '^ *[0-9]+ .*python .*experiments/eval/run.py'`
99. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 | sort`
100. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 | sort`
101. `bash -lc 'STAMP=20260401_000435; PRESERVE=reports/rc-t-012/_preserved_interrupted_${STAMP}; mkdir -p ${PRESERVE} && mv reports/rc-t-012/dow-paper-v1_ff5mom_w126 reports/rc-t-012/dow-paper-v1_noprewhiten_w126 ${PRESERVE}/ && echo ${PRESERVE}'`
102. `test ! -e reports/rc-t-012/dow-paper-v1_ff5mom_w126 && echo ABSENT`
103. `test ! -e reports/rc-t-012/dow-paper-v1_noprewhiten_w126 && echo ABSENT`
104. `find reports/rc-t-012/_preserved_interrupted_20260401_000435 -maxdepth 2 | sort`
105. `uptime`
106. `free -h`
107. `nproc`
108. `ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | head -n 20`
109. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
110. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
111. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
112. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_ff5mom_w126 --exec-mode deterministic'`
113. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_noprewhiten.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w126 --exec-mode deterministic'`
114. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
115. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
116. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
117. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
118. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
119. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
120. `tail -n 20 reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.log`
121. `tail -n 20 reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.log`
122. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
123. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
124. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
125. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w126|reports/rc-t-012/dow-paper-v1_noprewhiten_w126'`
126. `tail -n 20 reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.log`
127. `tail -n 20 reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.log`
128. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
129. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
130. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
131. `ps -eo pid,cmd | rg '^ *[0-9]+ python experiments/eval/run.py' | rg 'reports/rc-t-012'`
132. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.json`
133. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.json`
134. `tail -n 40 reports/rc-t-012/dow-paper-v1_ff5mom_w126/run.log`
135. `tail -n 40 reports/rc-t-012/dow-paper-v1_noprewhiten_w126/run.log`
136. `find reports/rc-t-012/dow-paper-v1_ff5mom_w126 -maxdepth 2 -type f | sort`
137. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w126 -maxdepth 2 -type f | sort`
138. `uptime`
139. `free -h`
140. `nproc`
141. `ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | head -n 20`
142. `test ! -e reports/rc-t-012/dow-paper-v1_ff5mom_w252 && echo ABSENT`
143. `test ! -e reports/rc-t-012/dow-paper-v1_noprewhiten_w252 && echo ABSENT`
144. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
145. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
146. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
147. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_window252.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_ff5mom_w252 --exec-mode deterministic'`
148. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python experiments/eval/run.py --config experiments/eval/config.paper_v1_dow_window252_noprewhiten.yaml --returns-csv data/returns_daily.csv --factors-csv data/factors/ff5mom_daily.csv --out ${RC_ROOT}/dow-paper-v1_noprewhiten_w252 --exec-mode deterministic'`
149. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w252|reports/rc-t-012/dow-paper-v1_noprewhiten_w252'`
150. `find reports/rc-t-012/dow-paper-v1_ff5mom_w252 -maxdepth 2 -type f | sort`
151. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w252 -maxdepth 2 -type f | sort`
152. `date -u '+%Y-%m-%d %H:%M:%S UTC'`
153. `ps -eo pid,etime,pcpu,pmem,cmd | rg 'reports/rc-t-012/dow-paper-v1_ff5mom_w252|reports/rc-t-012/dow-paper-v1_noprewhiten_w252'`
154. `tail -n 20 reports/rc-t-012/dow-paper-v1_ff5mom_w252/run.log`
155. `tail -n 20 reports/rc-t-012/dow-paper-v1_noprewhiten_w252/run.log`
156. `find reports/rc-t-012/dow-paper-v1_ff5mom_w252 -maxdepth 2 -type f | sort`
157. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w252 -maxdepth 2 -type f | sort`
158. `cat reports/rc-t-012/dow-paper-v1_ff5mom_w252/run.json`
159. `cat reports/rc-t-012/dow-paper-v1_noprewhiten_w252/run.json`
160. `sed -n '1,240p' docs/tickets/T-012-execute-the-next-daily-dow-empirical-campaign.md`
161. `rg -n "campaign_decision|full_regime_comparison|make_summary.py|reports/rc-t-012" -S docs reports scripts tools PROGRESS.md project_state docs/plan`
162. `rg --files reports | rg 'campaign_decision\.md|full_regime_comparison\.csv|summary\.csv|overlay_toggle\.md|regime_comparison'`
163. `sed -n '1,200p' reports/rc-t-010/summary/advisor_decision.md`
164. `sed -n '1,40p' reports/rc-t-010/summary/t010_full_regime_comparison.csv`
165. `sed -n '1,200p' reports/_runs/20260331_014033_t-010/COMMANDS.md`
166. `sed -n '1,220p' reports/_runs/20260331_014033_t-010/TESTS.md`
167. `sed -n '1,220p' docs/plan/NOW.md`
168. `sed -n '1,220p' project_state/CURRENT_STATE.md`
169. `sed -n '1,220p' project_state/KNOWN_ISSUES.md`
170. `sed -n '1,260p' reports/_runs/20260331_230430_t-012/COMMANDS.md`
171. `sed -n '1,320p' reports/_runs/20260331_230430_t-012/RESULTS.md`
172. `sed -n '1,320p' reports/_runs/20260331_230430_t-012/TESTS.md`
173. `sed -n '1,20p' reports/rc-t-010/summary/summary_perf.csv`
174. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_ff5mom_w126/full/metrics.csv`
175. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_noprewhiten_w126/full/metrics.csv`
176. `python - <<'PY' ... print reports/rc-t-010/summary/summary_perf.csv column names ... PY`
177. `sed -n '1,120p' reports/rc-t-012/dow-paper-v1_ff5mom_w252/run.json`
178. `sed -n '1,120p' reports/rc-t-012/dow-paper-v1_noprewhiten_w252/run.json`
179. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_ff5mom_w252/run.json`
180. `ps -eo pid,cmd | rg '^ *[0-9]+ python experiments/eval/run.py' | rg 'reports/rc-t-012'`
181. `sed -n '1,160p' reports/rc-t-012/dow-paper-v1_noprewhiten_w252/run.json`
182. `tail -n 40 reports/rc-t-012/dow-paper-v1_noprewhiten_w252/run.log`
183. `find reports/rc-t-012/dow-paper-v1_noprewhiten_w252 -maxdepth 2 -type f | sort`
184. `bash -lc 'RC_ROOT=reports/rc-t-012; . .venv/bin/activate && PYTHONPATH=src:. python tools/make_summary.py --rc-dir ${RC_ROOT}'`
185. `sed -n '1,40p' reports/rc-t-012/summary/summary_perf.csv`
186. `cat reports/rc-t-012/summary/kill_criteria.json`
187. `cat reports/rc-t-012/summary/completeness.json`
188. `sed -n '1,120p' reports/rc-t-012/summary/summary_detection.csv`
189. `sed -n '1,120p' reports/rc-t-012/summary/overlay_forensics.csv`
190. `python - <<'PY' ... derive reports/rc-t-012/summary/t012_full_regime_comparison.csv and reports/rc-t-012/summary/campaign_decision.md from reports/rc-t-012/summary/summary_perf.csv and reports/rc-t-010/summary/t010_full_regime_comparison.csv ... PY`
191. `sed -n '1,40p' reports/rc-t-012/summary/t012_full_regime_comparison.csv`
192. `sed -n '1,240p' reports/rc-t-012/summary/campaign_decision.md`
193. `python - <<'PY' ... T-012 artifact and comparison assertion over reports/rc-t-012 ... PY`
194. `cat reports/_runs/20260331_230430_t-012/META.json`
195. `sed -n '1,80p' PROGRESS.md`
196. `rg -n '"status": "(completed|complete|partial|initialized|blocked)"' reports/_runs -S`
197. `sed -n '1,80p' reports/_runs/20260331_014033_t-010/META.json`
198. `sed -n '1,80p' reports/_runs/20260331_220201_t-011/META.json`
199. `sed -n '1,220p' docs/plan/NOW.md`
200. `sed -n '1,260p' project_state/KNOWN_ISSUES.md`
201. `sed -n '70,190p' project_state/CURRENT_STATE.md`
202. `date -u '+%Y%m%d_%H%M%S'`
203. `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
204. `python scripts/make_gpt_bundle.py --root . --ticket T-012 --run-dir reports/_runs/20260331_230430_t-012 --out reports/_bundles/20260401_055651_T-012_gpt_bundle.zip`
205. `python - <<'PY' ... bundle assertion over reports/_bundles/20260401_055651_T-012_gpt_bundle.zip ... PY`
206. `rg -n "advisor_decision|rc-t-010|campaign_decision|t012_full_regime_comparison|rc-t-012" scripts/make_gpt_bundle.py`
207. `sed -n '1,260p' scripts/make_gpt_bundle.py`
208. `bash -lc '. .venv/bin/activate && pytest -q tests/test_gpt_bundle.py'`
209. `python scripts/make_gpt_bundle.py --root . --ticket T-012 --run-dir reports/_runs/20260331_230430_t-012 --out reports/_bundles/20260401_055651_T-012_gpt_bundle.zip`
210. `python - <<'PY' ... bundle assertion over reports/_bundles/20260401_055651_T-012_gpt_bundle.zip ... PY`
211. `date -u '+%Y-%m-%d %H:%M UTC'`
212. `make -f Makefile.agentic validate-runlog RUN_DIR=reports/_runs/20260331_230430_t-012`
213. `python scripts/make_gpt_bundle.py --root . --ticket T-012 --run-dir reports/_runs/20260331_230430_t-012 --out reports/_bundles/20260401_055651_T-012_gpt_bundle.zip`
214. `python - <<'PY' ... final bundle assertion over reports/_bundles/20260401_055651_T-012_gpt_bundle.zip ... PY`
