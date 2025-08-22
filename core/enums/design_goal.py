from __future__ import annotations

from enum import Enum

from core.models.design_goal_tile import DesignGoalTile


class DesignGoalTiles(Enum):
    """The six unique design goal configurations.

    Uses the letter notation defined in docs/DESIGN_GOAL_TILES.md.
    """

    SIX_UNIQUE = DesignGoalTile(
        config_name="A-B-C-D-E-F", config_numbers=[1, 1, 1, 1, 1, 1], lower_points=10, higher_points=15
    )
    THREE_PAIRS = DesignGoalTile(config_name="AA-BB-CC", config_numbers=[2, 2, 2], lower_points=7, higher_points=11)
    TWO_TRIPLETS = DesignGoalTile(config_name="AAA-BBB", config_numbers=[3, 3], lower_points=8, higher_points=13)
    THREE_TWO_ONE = DesignGoalTile(config_name="AAA-BB-C", config_numbers=[3, 2, 1], lower_points=7, higher_points=11)
    TWO_TWO_ONE_ONE = DesignGoalTile(
        config_name="AA-BB-C-D", config_numbers=[2, 2, 1, 1], lower_points=5, higher_points=8
    )
    FOUR_TWO = DesignGoalTile(config_name="AAAA-BB", config_numbers=[4, 2], lower_points=8, higher_points=14)
