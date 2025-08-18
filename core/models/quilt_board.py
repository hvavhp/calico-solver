from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from core.enums.edge_tile_settings import EdgeTileSettings
from core.models.design_goal_tile import DesignGoalTile
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
    """A player's quilt board mapping hex positions to placed tiles.

    - Uses axial coordinates to index hexes
    - Automatically initializes with edge tiles and design goal tiles
    - Edge tiles are pre-filled based on the selected edge tile setting
    - Three design goal tiles are placed at fixed positions (4,3), (5,4), (3,5)
    """

    tiles_by_pos: dict[HexPosition, PatchTile | DesignGoalTile] = Field(
        default_factory=dict,
        description="Mapping from hex position to placed tiles (patch tiles or design goal tiles).",
    )

    design_goal_tiles: list[DesignGoalTile] = Field(
        default_factory=list,
        description="List of the three design goal tiles for this board.",
    )

    edge_setting: EdgeTileSettings = Field(
        default=EdgeTileSettings.RAINBOW_STRIPES,
        description="The edge tile configuration used for this board.",
    )

    @field_validator("tiles_by_pos")
    @classmethod
    def _no_none_tiles(
        cls, value: dict[HexPosition, PatchTile | DesignGoalTile | None]
    ) -> dict[HexPosition, PatchTile | DesignGoalTile]:
        for pos, tile in value.items():
            if tile is None:
                msg = f"Hex {pos} has None tile"
                raise ValueError(msg)
        return value  # type: ignore[return-value]

    @model_validator(mode="after")
    def _initialize_board(self) -> QuiltBoard:
        """Initialize the board with edge tiles and design goal tiles if not already set."""
        # Only initialize if the board is empty (to avoid overriding existing configurations)
        edge_tiles_config = self.edge_setting.get_edge_tiles()
        for (q, r), patch_tile in edge_tiles_config.items():
            hex_pos = HexPosition(q=q, r=r)
            self.tiles_by_pos[hex_pos] = patch_tile

        self.set_design_goals(self.design_goal_tiles)
        return self

    def set_design_goals(self, goal_tiles: list[DesignGoalTile]) -> None:
        """Set the three design goal tiles and place them at fixed positions.

        Args:
            goal_tiles: List of exactly 3 DesignGoalTile objects

        Raises:
            ValueError: If the number of goal tiles is not exactly 3
        """
        if len(goal_tiles) != 3:
            msg = f"Expected exactly 3 design goal tiles, got {len(goal_tiles)}"
            raise ValueError(msg)

        self.design_goal_tiles = goal_tiles.copy()

        # Add design goal tiles to fixed positions
        design_goal_positions = [HexPosition(q=4, r=3), HexPosition(q=5, r=4), HexPosition(q=3, r=5)]

        for pos, goal_tile in zip(design_goal_positions, self.design_goal_tiles, strict=True):
            self.tiles_by_pos[pos] = goal_tile
