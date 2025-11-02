from __future__ import annotations

import pytest

from experiments.eval.run import EvalConfig, main, parse_args


def test_eval_parse_args_defaults() -> None:
    config = parse_args([])
    assert isinstance(config, EvalConfig)
    assert config.regime == "full"


def test_eval_main_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        main(["--regime", "crisis"])
