from __future__ import annotations

import pytest

from scripts.calibrate_thresholds import main, parse_args


def test_calibration_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.p == 200
    assert args.n == 252


def test_calibration_main_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        main(["--p", "100"])
