from __future__ import annotations

from enum import Enum


class Color(str, Enum):
    """The six distinct colors used by Calico patch tiles and button scoring.

    Names chosen to be human-friendly and stable for serialization.
    """

    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"


ALL_COLORS: tuple[Color, ...] = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.ORANGE,
    Color.PURPLE,
    Color.PINK,
)
