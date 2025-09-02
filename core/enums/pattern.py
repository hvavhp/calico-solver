from __future__ import annotations

from enum import Enum


class Pattern(str, Enum):
    """The six distinct patterns used by Calico patch tiles and cat scoring.

    Pattern names mirror common community usage and are stable for serialization.
    """

    STRIPES = "stripes"
    DOTS = "dots"
    FLOWERS = "flowers"
    VINES = "vines"
    QUATREFOIL = "quatrefoil"
    FERNS = "ferns"


ALL_PATTERNS: tuple[Pattern, ...] = (
    Pattern.STRIPES,
    Pattern.DOTS,
    Pattern.FLOWERS,
    Pattern.VINES,
    Pattern.QUATREFOIL,
    Pattern.FERNS,
)

PATTERN_MAP: dict[Pattern, int] = {pattern: i for i, pattern in enumerate(ALL_PATTERNS)}
