from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DesignGoalTile(BaseModel):
    """A design goal tile evaluated on the six neighbors around its board spot.

    - config: one of the six unique letter-notation configurations
    - lower_points: points for satisfying by color OR pattern
    - higher_points: points for satisfying by color AND pattern simultaneously
    - by_attribute: used when referencing how a particular satisfaction was met
    """

    config: str
    lower_points: int = Field(..., ge=0)
    higher_points: int = Field(..., ge=0)

    # Optional helper to annotate an evaluation result externally
    by_attribute: Literal["color", "pattern", "both"] | None = None
