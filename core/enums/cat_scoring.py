from __future__ import annotations

from enum import Enum
from re import Pattern

from models.cat_scoring_tile import (
    CatDifficultyGroup,
    CatScoringTile,
    CatShapeType,
    GroupSizeRequirement,
    ShapeRequirement,
)


class CatScoringConfig(str, Enum):
    """The six unique design goal configurations.

    Uses the letter notation defined in docs/DESIGN_GOAL_TILES.md.
    """

    MILLIE = CatScoringTile(
        name="Millie",
        difficulty_group=CatDifficultyGroup.ONE_DOT,
        requirement=GroupSizeRequirement(min_size=3),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[3],
    )
    TIBBIT = CatScoringTile(
        name="Tibbit",
        difficulty_group=CatDifficultyGroup.ONE_DOT,
        requirement=GroupSizeRequirement(min_size=4),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[5],
    )
    COCONUT = CatScoringTile(
        name="Coconut",
        difficulty_group=CatDifficultyGroup.TWO_DOT,
        requirement=GroupSizeRequirement(min_size=5),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[7],
    )
    CIRA = CatScoringTile(
        name="Cira",
        difficulty_group=CatDifficultyGroup.TWO_DOT,
        requirement=GroupSizeRequirement(min_size=6),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[9],
    )
    GWENIVERE = CatScoringTile(
        name="Gwenivere",
        difficulty_group=CatDifficultyGroup.THREE_DOT,
        requirement=GroupSizeRequirement(min_size=7),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[11],
    )
    CALLIE = CatScoringTile(
        name="Callie",
        difficulty_group=CatDifficultyGroup.ONE_DOT,
        requirement=ShapeRequirement(shape=CatShapeType.TRIANGLE_3),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[3],
    )
    RUMI = CatScoringTile(
        name="Rumi",
        difficulty_group=CatDifficultyGroup.ONE_DOT,
        requirement=ShapeRequirement(shape=CatShapeType.LINE_3),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[5],
    )
    TECOLOTE = CatScoringTile(
        name="Tecolote",
        difficulty_group=CatDifficultyGroup.TWO_DOT,
        requirement=ShapeRequirement(shape=CatShapeType.LINE_4),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[7],
    )
    ALMOND = CatScoringTile(
        name="Almond",
        difficulty_group=CatDifficultyGroup.TWO_DOT,
        requirement=ShapeRequirement(shape=CatShapeType.T_SHAPE_5),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[9],
    )
    LEO = CatScoringTile(
        name="Leo",
        difficulty_group=CatDifficultyGroup.THREE_DOT,
        requirement=ShapeRequirement(shape=CatShapeType.LINE_5),
        allowed_patterns=(Pattern.STRIPES, Pattern.POLKA_DOTS),
        token_values_desc=[11],
    )
