from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from core.models.patch_tile import PatchTile


class HexPosition(BaseModel):
    """Axial hex coordinates using q (column), r (row).

    The project docs reference a 7x7 board with 3 fixed goal positions at
    (4,3), (5,4), and (3,5). We keep coordinates generic and let higher-level
    rules enforce bounds/occupancy.
    """

    q: int = Field(..., description="Column index (axial q)")
    r: int = Field(..., description="Row index (axial r)")

    model_config = {
        "frozen": True,
    }


class QuiltBoard(BaseModel):
    """A player's quilt board mapping hex positions to placed patch tiles.

    - Uses axial coordinates to index hexes
    - No game logic beyond shape; validation stays minimal here
    """

    tiles_by_pos: dict[HexPosition, PatchTile] = Field(
        default_factory=dict,
        description="Mapping from hex position to placed patch tile.",
    )

    @field_validator("tiles_by_pos")
    @classmethod
    def _no_none_tiles(cls, value: dict[HexPosition, PatchTile | None]) -> dict[HexPosition, PatchTile]:
        for pos, tile in value.items():
            if tile is None:
                msg = f"Hex {pos} has None tile"
                raise ValueError(msg)
        return value  # type: ignore[return-value]
