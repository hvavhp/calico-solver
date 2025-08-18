from __future__ import annotations

from enum import Enum


class Pattern(str, Enum):
    """The six distinct patterns used by Calico patch tiles and cat scoring.

    Pattern names mirror common community usage and are stable for serialization.
    """

    STRIPES = "stripes"
    POLKA_DOTS = "polka_dots"
    FLORAL = "floral"
    VINES = "vines"
    PLAID = "plaid"
    HERRINGBONE = "herringbone"


ALL_PATTERNS: tuple[Pattern, ...] = (
    Pattern.STRIPES,
    Pattern.POLKA_DOTS,
    Pattern.FLORAL,
    Pattern.VINES,
    Pattern.PLAID,
    Pattern.HERRINGBONE,
)
