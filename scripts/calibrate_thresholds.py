"""
Synthetic calibration harness scaffolding.

Future milestones implement threshold sweeps that enforce FPR ≤2% at matched p/n.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for calibration."""

    parser = argparse.ArgumentParser(description="Calibrate detection thresholds (placeholder).")
    parser.add_argument("--p", type=int, default=200, help="Dimensionality of the synthetic panel.")
    parser.add_argument("--n", type=int, default=252, help="Sample size per group.")
    parser.add_argument("--mu", type=str, default="4,6,8", help="Signal strengths to sweep.")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/calibration/",
        help="Output directory for ROC curves and thresholds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Placeholder entry point until the calibration harness is implemented."""

    _ = parse_args(argv)
    raise NotImplementedError("Calibration harness will be implemented in Sprint 1.")


if __name__ == "__main__":
    main()
