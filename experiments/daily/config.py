from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DailyDesign", "DAILY_DESIGNS"]


@dataclass(frozen=True)
class DailyDesign:
    """Configuration defaults for a replicated daily experiment."""

    name: str
    title: str
    group_design: str
    edge_mode: str
    group_min_count: int
    group_min_replicates: int
    prewhiten: str


DAILY_DESIGNS: dict[str, DailyDesign] = {
    "dow": DailyDesign(
        name="dow",
        title="Day-of-Week Replicates",
        group_design="dow",
        edge_mode="tyler",
        group_min_count=5,
        group_min_replicates=3,
        prewhiten="ff5mom",
    ),
    "vol": DailyDesign(
        name="vol",
        title="Volatility-State Replicates",
        group_design="vol",
        edge_mode="huber",
        group_min_count=3,
        group_min_replicates=4,
        prewhiten="ff5mom",
    ),
}
