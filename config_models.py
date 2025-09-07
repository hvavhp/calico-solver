"""
Pydantic models for configuration management in the Calico solver.

This module defines the data structures for reading JSON configuration files
that specify multiple optimization configurations to run in parallel.
"""

from typing import Literal

from pydantic import BaseModel, Field

from core.enums.design_goal import DesignGoalTiles


class OptimizationConfiguration(BaseModel):
    """
    A single optimization configuration specifying design goals and cats.

    This represents one test case to be run by the optimization engine.
    """

    # Design goals (order matters - represents the 3 design goal positions)
    design_goals: list[
        Literal["SIX_UNIQUE", "THREE_PAIRS", "TWO_TRIPLETS", "THREE_TWO_ONE", "TWO_TWO_ONE_ONE", "FOUR_TWO"]
    ] = Field(..., description="List of 3 design goal names in order", min_items=3, max_items=3)

    # Cats (order doesn't matter - just the 3 cats to use)
    cats: list[
        Literal["millie", "tibbit", "coconut", "cira", "gwenivere", "callie", "rumi", "tecolote", "almond", "leo"]
    ] = Field(..., description="List of 3 cat names", min_items=3, max_items=3)

    def get_design_goal_tiles(self):
        """Convert design goal names to DesignGoalTiles enum values."""
        return tuple(getattr(DesignGoalTiles, name).value for name in self.design_goals)
