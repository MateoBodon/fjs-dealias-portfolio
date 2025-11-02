"""
Rolling evaluation harness scaffolding.

The final version will compute ΔMSE, VaR/ES, and Diebold–Mariano statistics
across regimes.  This placeholder provides the CLI structure for upcoming work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class EvalConfig:
    """Configuration placeholder for the rolling evaluation."""

    regime: str = "full"
    output_dir: str = "reports/rc-latest"


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    """Parse CLI arguments into an ``EvalConfig``."""

    parser = argparse.ArgumentParser(description="Rolling evaluation placeholder.")
    parser.add_argument("--regime", default="full", help="Regime to evaluate (placeholder).")
    parser.add_argument(
        "--out",
        dest="output_dir",
        default="reports/rc-latest",
        help="Output directory for evaluation artifacts.",
    )
    args = parser.parse_args(argv)
    return EvalConfig(regime=args.regime, output_dir=args.output_dir)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the rolling evaluation (to be implemented in Sprint 2)."""

    _ = parse_args(argv)
    raise NotImplementedError("Rolling evaluation will be implemented in Sprint 2.")


if __name__ == "__main__":
    main()
