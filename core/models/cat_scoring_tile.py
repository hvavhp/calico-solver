from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from core.enums import Pattern


class CatDifficultyGroup(int, Enum):
    """Dot difficulty groups used during setup selection."""

    ONE_DOT = 1
    TWO_DOT = 2
    THREE_DOT = 3


class CatShapeType(str, Enum):
    """Known shape-based requirements in Calico (see docs/CAT_SCORING_TILES.md)."""

    TRIANGLE_3 = "triangle_3"
    LINE_3 = "line_3"
    LINE_4 = "line_4"
    T_SHAPE_5 = "t_shape_5"
    LINE_5 = "line_5"


class GroupSizeRequirement(BaseModel):
    """Size-based requirement: contiguous group of one pattern with size >= n."""

    min_size: int = Field(..., ge=3, description="Minimum contiguous group size (>=3)")


class ShapeRequirement(BaseModel):
    """Shape-based requirement: exact arrangement in one pattern.

    The specific shape encoding is abstracted as an enum here.
    """

    shape: CatShapeType


class CatScoringTile(BaseModel):
    """A cat scoring tile with requirement and two allowed patterns.

    - name: the cat tile name (e.g., Millie, Rumi)
    - difficulty_group: one of the three dot groups (1, 2, or 3 dots)
    - requirement: either a GroupSizeRequirement or ShapeRequirement
    - allowed_patterns: exactly two patterns assigned during setup
    - token_values_desc: ordered high-to-low stack of token values for this cat
    """

    name: str
    difficulty_group: CatDifficultyGroup
    requirement: GroupSizeRequirement | ShapeRequirement
    allowed_patterns: tuple[Pattern, Pattern]
    token_values_desc: list[int] = Field(
        default_factory=list,
        description="Highest to lowest remaining values for this cat's tokens.",
    )
