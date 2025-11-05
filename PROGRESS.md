## 2025-11-05T21:52Z — sprint-1 calibration & smoke (2d235955)
- **Data**: `data/returns_daily.csv` (`sha1=1ff062eab6f0741f7fdc8d25098ffb8f9e3a5344`)
- **Commands**: `make sweep:acceptance`, `make run:equity_smoke`, `make memo`, `make test`
- **Artifacts**: `reports/figures/roc_null.png`, `reports/figures/roc_power.png`, `calibration_defaults.json`, `reports/synthetic/{null_harness,power_harness}/`, `experiments/equity_panel/outputs_smoke/`, `reports/memo.md`
- **Notes**: Synthetic harness writes score tables + calibration defaults (energy floor), detection summary now logs edge bands/gating mode and MV solver stats, MV defaults locked to ridge=1e-4 with box [0,0.1] and 5 bps turnover cost cap.
