from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval.run import EvalConfig, run_evaluation


@dataclass(slots=True, frozen=True)
class ETFConfig:
    returns_csv: Path
    factors_csv: Path | None
    out_dir: Path
    window: int = 252
    horizon: int = 21
    shrinker: str = "rie"
    seed: int = 42


def parse_args(argv: Sequence[str] | None = None) -> ETFConfig:
    parser = argparse.ArgumentParser(description="ETF alt-panel evaluation demo.")
    parser.add_argument(
        "--returns-csv",
        type=Path,
        default=Path("data/etf_returns.csv"),
        help="Daily ETF returns CSV (date plus tickers).",
    )
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional factor CSV to prewhiten returns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/etf-eval"),
        help="Output directory for ETF diagnostics.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Estimation window in trading days (default: one year).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Holdout horizon in trading days (default: one month).",
    )
    parser.add_argument(
        "--shrinker",
        type=str,
        default="rie",
        choices=["rie", "lw", "oas", "sample"],
        help="Shrinkage model for non-detected eigenvalues.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    args = parser.parse_args(argv)
    return ETFConfig(
        returns_csv=args.returns_csv,
        factors_csv=args.factors_csv,
        out_dir=args.out,
        window=args.window,
        horizon=args.horizon,
        shrinker=args.shrinker,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    eval_cfg = EvalConfig(
        returns_csv=cfg.returns_csv,
        factors_csv=cfg.factors_csv,
        window=cfg.window,
        horizon=cfg.horizon,
        out_dir=cfg.out_dir,
        shrinker=cfg.shrinker,
        seed=cfg.seed,
    )
    run_evaluation(eval_cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
