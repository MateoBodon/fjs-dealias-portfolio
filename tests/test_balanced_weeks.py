import numpy as np
import pandas as pd

from finance.io import to_daily_returns
from finance.returns import balance_weeks


def test_balance_weeks_drops_partial_periods() -> None:
    tickers = ["AAA", "BBB"]
    seed_prices = {"AAA": 100.0, "BBB": 55.0}

    day_sequence = [
        pd.Timestamp("2022-12-30"),
        *pd.date_range("2023-01-02", "2023-01-06", freq="B"),
        pd.Timestamp("2023-01-09"),
        pd.Timestamp("2023-01-10"),
        pd.Timestamp("2023-01-12"),
        pd.Timestamp("2023-01-13"),
        *pd.date_range("2023-01-16", "2023-01-20", freq="B"),
        *pd.date_range("2023-01-23", "2023-01-27", freq="B"),
    ]

    rows: list[dict[str, object]] = []
    for ticker in tickers:
        price = seed_prices[ticker]
        for offset, date in enumerate(day_sequence, start=1):
            price += 0.5 + 0.1 * offset
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "price_close": price + (0.2 if ticker == "BBB" else 0.0),
                }
            )

    price_frame = pd.DataFrame(rows).sort_values(["date", "ticker"])
    daily_returns = to_daily_returns(price_frame)

    y_matrix, groups, week_index = balance_weeks(daily_returns)

    unique_groups, counts = np.unique(groups, return_counts=True)
    assert np.all(counts == 5)
    assert len(unique_groups) == 2
    assert len(week_index) == 2
    expected_weeks = pd.to_datetime(["2023-01-03", "2023-01-17"])
    pd.testing.assert_index_equal(week_index, pd.DatetimeIndex(expected_weeks))
    assert y_matrix.shape[0] == 5 * len(unique_groups)
