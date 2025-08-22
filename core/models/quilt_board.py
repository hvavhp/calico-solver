from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from core.enums.color import Color
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import Pattern
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

    @property
    def abs(self):
        return self.q + self.r * 7


class QuiltBoard(BaseModel):
    """A player's quilt board mapping hex positions to placed tiles.

    - Uses axial coordinates to index hexes
    - Automatically initializes with edge tiles and design goal tiles
    - Edge tiles are pre-filled based on the selected edge tile setting
    - Three design goal tiles are placed at fixed positions (4,3), (5,4), (3,5)
    """

    tiles_by_pos: dict[HexPosition, PatchTile | DesignGoalTile] | None = Field(
        default_factory=dict,
        description="Mapping from hex position to placed tiles (patch tiles or design goal tiles).",
    )

    design_goal_tiles: list[DesignGoalTile] | None = Field(
        default_factory=list,
        description="List of the three design goal tiles for this board.",
    )

    edge_setting: EdgeTileSettings | None = Field(
        default=EdgeTileSettings.BOARD_1,
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
        design_goal_positions = self.get_design_goal_tile_positions()

        for pos, goal_tile in zip(design_goal_positions, self.design_goal_tiles, strict=True):
            self.tiles_by_pos[pos] = goal_tile

    def _get_hex_neighbors(self, pos: HexPosition) -> list[HexPosition]:
        """Get all 6 potential neighbors of a hexagonal tile.

        In this hexagonal grid, rows 0,2,4,6 are aligned, while rows 1,3,5 are shifted right.
        This requires different neighbor calculations for even vs odd rows.

        Args:
            pos: The hex position to find neighbors for

        Returns:
            List of neighboring hex positions (may include positions outside board bounds)
        """
        q, r = pos.q, pos.r

        if r % 2 == 0:  # Even rows (0, 2, 4, 6) - aligned
            neighbor_offsets = [
                (-1, -1),  # northwest
                (0, -1),  # northeast
                (-1, 0),  # east
                (1, 0),  # west
                (-1, 1),  # southeast
                (0, 1),  # southwest
            ]
        else:  # Odd rows (1, 3, 5) - shifted right
            neighbor_offsets = [
                (0, -1),  # northwest
                (1, -1),  # northeast
                (-1, 0),  # east
                (1, 0),  # west
                (0, 1),  # southeast
                (1, 1),  # southwest
            ]

        neighbors = []
        for dq, dr in neighbor_offsets:
            if 0 <= q + dq <= 6 and 0 <= r + dr <= 6:
                neighbors.append(HexPosition(q=q + dq, r=r + dr))

        return neighbors

    def _is_edge_position(self, pos: HexPosition) -> bool:
        """Check if a position is an edge tile position.

        Args:
            pos: The position to check

        Returns:
            True if the position is an edge tile
        """
        q, r = pos.q, pos.r

        # Check if position is on the border of the 7x7 board
        return q == 0 or q == 6 or r == 0 or r == 6

    def _is_design_goal_position(self, pos: HexPosition) -> bool:
        """Check if a position is a design goal tile position.

        Args:
            pos: The position to check

        Returns:
            True if the position is a design goal tile
        """
        design_goal_positions = [HexPosition(q=3, r=2), HexPosition(q=4, r=3), HexPosition(q=2, r=4)]
        return pos in design_goal_positions

    def get_three_neighbor_tile_sets(self) -> list[list[HexPosition]]:
        """Get all possible sets of three neighboring tiles, excluding edge and design goal tiles.

        Returns:
            List of tile sets, where each set is a list of 3 neighboring HexPosition objects.
            Each set represents 3 tiles that are all connected to each other.
        """
        # Get all valid (non-edge, non-design-goal) positions
        valid_positions = []
        for q in range(7):
            for r in range(7):
                pos = HexPosition(q=q, r=r)
                if not self._is_design_goal_position(pos):
                    valid_positions.append(pos)

        three_tile_sets = []

        # For each valid position, find all triangular combinations with its neighbors
        for _i, pos1 in enumerate(valid_positions):
            neighbors = [n for n in self._get_hex_neighbors(pos1) if n in valid_positions]

            # For each pair of neighbors of pos1, check if they are also neighbors of each other
            for j, pos2 in enumerate(neighbors):
                for k, pos3 in enumerate(neighbors):
                    if j >= k:  # Avoid duplicates and self-comparison
                        continue

                    if self._is_edge_position(pos2) and self._is_edge_position(pos3):
                        continue

                    if self._is_edge_position(pos2) and self._is_edge_position(pos1):
                        continue

                    if self._is_edge_position(pos3) and self._is_edge_position(pos1):
                        continue

                    tile_set = sorted([pos1, pos2, pos3], key=lambda p: (p.q, p.r))
                    if tile_set not in three_tile_sets:
                        three_tile_sets.append(tile_set)

        return three_tile_sets

    def get_design_goal_tile_positions(self) -> list[HexPosition]:
        return [HexPosition(q=3, r=2), HexPosition(q=4, r=3), HexPosition(q=2, r=4)]

    def get_all_patch_tiles(self) -> list[HexPosition]:
        ret = []
        for q in range(7):
            for r in range(7):
                pos = HexPosition(q=q, r=r)
                if not self._is_edge_position(pos) and not self._is_design_goal_position(pos):
                    ret.append(pos)
        return ret

    def get_design_goal_patch_tiles(self, goal_tile_no: int | None = None) -> list[HexPosition]:
        design_goal_positions = self.get_design_goal_tile_positions()
        if goal_tile_no is not None:
            design_goal_positions = [design_goal_positions[goal_tile_no]]
        neighbors = [self._get_hex_neighbors(tile) for tile in design_goal_positions]
        neighbors = [n for neighbor_list in neighbors for n in neighbor_list]
        neighbors = list(set(neighbors))
        return neighbors

    def get_two_neighbor_tile_sets_near_edge(self) -> list[list[HexPosition]]:
        """Get all pairs of neighboring tiles where at least one tile is adjacent to an edge tile.

        Returns:
            List of tile pairs, where each pair is a list of 2 neighboring HexPosition objects.
            At least one tile in each pair is adjacent to an edge tile.
        """

        valid_positions = []
        for q in range(7):
            for r in range(7):
                pos = HexPosition(q=q, r=r)
                if not self._is_edge_position(pos) and not self._is_design_goal_position(pos):
                    valid_positions.append(pos)

        # Find all edge tiles
        edge_tiles = []
        for q in range(7):
            for r in range(7):
                pos = HexPosition(q=q, r=r)
                if self._is_edge_position(pos):
                    edge_tiles.append(pos)

        # Find all tiles that are neighbors of edge tiles
        tiles_near_edge = set()
        for edge_tile in edge_tiles:
            neighbors = self._get_hex_neighbors(edge_tile)
            for neighbor in neighbors:
                # Only include neighbors within board bounds and not edge tiles themselves
                if not self._is_edge_position(neighbor):
                    tiles_near_edge.add(neighbor)

        # Find pairs of neighboring tiles where at least one is near an edge
        two_tile_sets = []

        # Convert set to list for easier iteration
        tiles_near_edge_list = list(tiles_near_edge)

        # Check all possible pairs
        for _i, pos1 in enumerate(tiles_near_edge_list):
            neighbors1 = self._get_hex_neighbors(pos1)

            for neighbor in neighbors1:
                # Skip if neighbor is an edge tile
                if self._is_edge_position(neighbor):
                    continue
                if self._is_design_goal_position(neighbor):
                    continue

                # Create the pair, ensuring consistent ordering
                tile_pair = sorted([pos1, neighbor], key=lambda p: (p.q, p.r))

                # Check if this pair is already in our results
                if tile_pair not in two_tile_sets:
                    two_tile_sets.append(tile_pair)

        return two_tile_sets

    def _get_color_symbol(self, color: Color) -> str:
        """Get a single character symbol for a color."""
        color_symbols = {
            Color.BLUE: "B",
            Color.GREEN: "G",
            Color.YELLOW: "Y",
            Color.NAVY: "N",
            Color.PURPLE: "P",
            Color.PINK: "K",  # K for pinK to avoid confusion with Purple
        }
        return color_symbols[color]

    def _get_pattern_symbol(self, pattern: Pattern) -> str:
        """Get a single character symbol for a pattern."""
        pattern_symbols = {
            Pattern.STRIPES: "▣ ",  # Striped square
            Pattern.DOTS: "● ",  # Dots
            Pattern.FLOWERS: "✿ ",  # Flower
            Pattern.VINES: "🌿",  # Vine/leaf
            Pattern.QUATREFOIL: "❋ ",  # Four-leaf design
            Pattern.FERNS: "🌾",  # Fern
        }
        return pattern_symbols[pattern]

    def _get_tile_display(self, pos: HexPosition) -> str:
        """Get display representation for a tile at the given position."""
        tile = self.tiles_by_pos.get(pos)

        # Check if it's a design goal position
        design_goal_positions = self.get_design_goal_tile_positions()
        if pos in design_goal_positions:
            if isinstance(tile, DesignGoalTile):
                goal_idx = design_goal_positions.index(pos)
                return f"G{goal_idx + 1}".ljust(3)  # Ensure consistent width
            return "G? ".ljust(3)  # Should not happen

        if isinstance(tile, PatchTile):
            color_sym = self._get_color_symbol(tile.color)
            pattern_sym = self._get_pattern_symbol(tile.pattern)
            return f"{color_sym}{pattern_sym}"  # Ensure consistent width

        # Empty space (no tile placed) - use a placeholder
        return "---"  # Clear indication of empty space

    def pretty_print(self) -> str:
        """Pretty-print the hexagonal quilt board with colors, patterns, and design goals.

        Shows:
        - Edge tiles with their actual color and pattern symbols
        - Design goal tiles as G1, G2, G3
        - Patch tiles with color + pattern symbols
        - Empty spaces for unfilled positions
        - Hexagonal layout with odd rows shifted right
        """
        lines = []

        # Add header
        lines.append("Calico Quilt Board (7x7 Hexagonal)")
        lines.append("Legend: [Color][Pattern] | G1,G2,G3=Design Goals | ---=Empty | Edge tiles shown")
        lines.append("Colors: B=Blue, G=Green, Y=Yellow, N=Navy, P=Purple, K=Pink")
        lines.append("Note: Odd rows (1,3,5) are shifted right to show hexagonal layout")
        lines.append("")

        # Build the hexagonal grid
        for r in range(7):
            line_parts = []

            # Add indentation for odd rows (hexagonal offset)
            if r % 2 == 1:
                line_parts.append("   ")  # Half-tile offset for odd rows (3 spaces to align properly)

            # Add tiles for this row
            for q in range(7):
                pos = HexPosition(q=q, r=r)
                tile_display = self._get_tile_display(pos)
                line_parts.append(tile_display)

                # Add spacing between tiles (consistent spacing)
                if q < 6:
                    line_parts.append("   ")  # 2 spaces between tiles

            # Add row number for reference
            line_parts.append(f"  (row {r})")

            lines.append("".join(line_parts))

        # Add column reference
        lines.append("")
        # Adjust column labels to align with tiles
        col_labels = []
        col_labels.append("(0) ")  # First column
        for i in range(1, 7):
            col_labels.append(f" ({i}) ")  # Subsequent columns with padding
        lines.append("".join(col_labels) + "  (columns)")

        return "\n".join(lines)
