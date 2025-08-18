from __future__ import annotations

from enum import Enum

from models.design_goal_tile import DesignGoalTile


class DesignGoalConfig(str, Enum):
    """The six unique design goal configurations.

    Uses the letter notation defined in docs/DESIGN_GOAL_TILES.md.
    """

    SIX_UNIQUE = DesignGoalTile(config="A-B-C-D-E-F", lower_points=10, higher_points=15)
    THREE_PAIRS = DesignGoalTile(config="AA-BB-CC", lower_points=7, higher_points=11)
    TWO_TRIPLETS = DesignGoalTile(config="AAA-BBB", lower_points=8, higher_points=13)
    THREE_TWO_ONE = DesignGoalTile(config="AAA-BB-C", lower_points=7, higher_points=11)
    TWO_TWO_ONE_ONE = DesignGoalTile(config="AA-BB-C-D", lower_points=5, higher_points=8)
    FOUR_TWO = DesignGoalTile(config="AAAA-BB", lower_points=8, higher_points=14)
