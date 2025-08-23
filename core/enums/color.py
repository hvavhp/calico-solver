from __future__ import annotations

from enum import Enum


class Color(str, Enum):
    """The six distinct colors used by Calico patch tiles and button scoring.

    Names chosen to be human-friendly and stable for serialization.
    """

    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    NAVY = "navy"
    PURPLE = "purple"
    PINK = "pink"


ALL_COLORS: tuple[Color, ...] = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.NAVY,
    Color.PURPLE,
    Color.PINK,
)

COLOR_MAP = {
    Color.BLUE: 0,
    Color.GREEN: 1,
    Color.YELLOW: 2,
    Color.NAVY: 3,
    Color.PURPLE: 4,
    Color.PINK: 5,
}
