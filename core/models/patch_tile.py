from __future__ import annotations

from pydantic import BaseModel

from core.enums.color import Color
from core.enums.pattern import Pattern


class PatchTile(BaseModel):
    """A hex patch tile with exactly one color and one pattern."""

    color: Color
    pattern: Pattern
