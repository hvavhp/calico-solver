from __future__ import annotations

from pydantic import BaseModel

from core.enums import Color, Pattern


class PatchTile(BaseModel):
    """A hex patch tile with exactly one color and one pattern."""

    color: Color
    pattern: Pattern
