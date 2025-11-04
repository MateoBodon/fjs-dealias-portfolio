from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from experiments.daily.config import DAILY_DESIGNS
from experiments.eval.run import main as eval_main

__all__ = ["main"]


def _detect_forward_override(argv: Sequence[str], flag: str) -> bool:
    flag_eq = f"{flag}="
    for idx, token in enumerate(argv):
        if token == flag:
            return True
        if token.startswith(flag_eq):
            return True
        if token.startswith("--") and token[2:] == flag.lstrip("--"):
            return True
        if token == "--" and idx + 1 < len(argv) and argv[idx + 1] == flag:
            return True
    return False


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser("Daily replicated RC runner", add_help=True)
    parser.add_argument("--returns-csv", type=Path, required=True, help="Daily returns CSV (wide or long format).")
    parser.add_argument("--design", choices=sorted(DAILY_DESIGNS.keys()), required=True, help="Replicate design to run.")
    parser.add_argument("--out", type=Path, default=None, help="Optional explicit output directory.")
    parser.add_argument("--factors-csv", type=Path, default=None, help="Optional factor CSV for prewhitening.")
    parser.add_argument("--window", type=int, default=None, help="Optional window override (days).")
    parser.add_argument("--horizon", type=int, default=None, help="Optional horizon override (days).")
    parser.add_argument("--shrinker", type=str, default=None, help="Baseline shrinker override for overlay baseline.")
    parser.add_argument("--rc-date", type=str, default=None, help="RC date stamp (defaults to today if omitted).")
    parser.add_argument(
        "--prewhiten",
        type=str,
        choices=["off", "ff5", "ff5mom"],
        default=None,
        help="Observed-factor prewhitening mode passed through to evaluation.",
    )
    args, extra = parser.parse_known_args(argv)
    return args, list(extra)


def _default_out(design: str, rc_date: str | None) -> Path:
    if not rc_date:
        rc_date = datetime.utcnow().strftime("%Y%m%d")
    return Path("reports") / f"rc-{rc_date}" / design


def main(argv: Sequence[str] | None = None) -> None:
    args, extra = parse_args(argv)
    design = DAILY_DESIGNS[args.design]

    forwarded: list[str] = ["--returns-csv", str(args.returns_csv)]

    if args.factors_csv is not None and not _detect_forward_override(extra, "--factors-csv"):
        forwarded.extend(["--factors-csv", str(args.factors_csv)])
    if args.window is not None and not _detect_forward_override(extra, "--window"):
        forwarded.extend(["--window", str(args.window)])
    if args.horizon is not None and not _detect_forward_override(extra, "--horizon"):
        forwarded.extend(["--horizon", str(args.horizon)])
    if args.shrinker and not _detect_forward_override(extra, "--shrinker"):
        forwarded.extend(["--shrinker", args.shrinker])
    if args.prewhiten and not _detect_forward_override(extra, "--prewhiten"):
        forwarded.extend(["--prewhiten", args.prewhiten])

    out_dir = args.out
    if out_dir is None and not _detect_forward_override(extra, "--out"):
        out_dir = _default_out(design.name, args.rc_date)
    if out_dir is not None:
        forwarded.extend(["--out", str(out_dir)])

    forwarded.extend(
        [
            "--group-design",
            design.group_design,
            "--group-min-count",
            str(design.group_min_count),
            "--group-min-replicates",
            str(design.group_min_replicates),
            "--edge-mode",
            design.edge_mode,
        ]
    )

    forwarded.extend(extra)
    eval_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    main()
