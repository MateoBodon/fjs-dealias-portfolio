from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.equity_panel import run as equity_run

pytestmark = pytest.mark.integration


def _make_prices_csv(tmp_path: Path, weeks: int = 12, assets: int = 6) -> Path:
    rng = np.random.default_rng(7)
    days = weeks * 5
    dates = pd.bdate_range("2020-01-06", periods=days)
    market = rng.normal(scale=0.004, size=len(dates))
    rows: list[dict[str, object]] = []
    for j in range(assets):
        beta = rng.normal(scale=0.6)
        idio = rng.normal(scale=0.01, size=len(dates))
        rets = beta * market + idio
        prices = 100.0 * np.exp(np.cumsum(rets))
        for t, d in enumerate(dates):
            rows.append(
                {"date": d, "ticker": f"A{j:02d}", "price_close": float(prices[t])}
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return path


def test_equity_ablation_emits_summary(tmp_path: Path) -> None:
    prices_path = _make_prices_csv(tmp_path, weeks=12, assets=8)
    cfg_text = f"""
data_path: "{prices_path}"
start_date: "2020-01-06"
end_date: "2020-03-27"
window_weeks: 4
horizon_weeks: 1
output_dir: "{tmp_path}/outputs"
dealias_delta: 0.0
dealias_delta_frac: 0.03
dealias_eps: 0.03
stability_eta_deg: 0.4
signed_a: true
cs_drop_top_frac: 0.05
a_grid: 90
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)

    equity_run.run_experiment(
        cfg_path,
        sigma_ablation=False,
        crisis=None,
        delta_frac_override=None,
        signed_a_override=None,
        target_component_override=None,
        cs_drop_top_frac_override=None,
        progress_override=False,
        eps_override=None,
        a_grid_override=None,
        ablations=True,
    )

    out_dir = tmp_path / "outputs"
    run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    assert run_dirs, "No run directory produced"
    run_dir = run_dirs[0]
    assert (run_dir / "ablation_summary.csv").exists()
