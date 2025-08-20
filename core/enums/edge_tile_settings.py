from __future__ import annotations

from enum import Enum

from core.enums.color import Color
from core.enums.pattern import Pattern
from core.models.patch_tile import PatchTile


class EdgeTileSettings(Enum):
    """The four different edge tile configurations for the quilt board.

    Each setting pre-fills the 24 edge positions with specific color/pattern
    combinations to create different strategic starting conditions.
    """

    BOARD_1 = "board_1"
    """Edge tiles with all 6 colors in alternating stripe patterns."""

    BOARD_2 = "board_2"
    """Top/bottom edges use warm colors, left/right edges use cool colors."""

    BOARD_3 = "board_3"
    """Edge tiles emphasize pattern diversity with balanced color distribution."""

    BOARD_4 = "board_4"
    """Edge tiles arranged in perfect rotational symmetry around the board."""

    def get_edge_tiles(self) -> dict[tuple[int, int], PatchTile]:
        """Returns the edge tile configuration as position -> PatchTile mapping.

        Uses (q, r) axial coordinates where the 7x7 board spans from (0,0) to (6,6).
        Edge positions are those on the border of this coordinate space.
        """
        if self == EdgeTileSettings.BOARD_1:
            return self._board_1_config()
        if self == EdgeTileSettings.BOARD_2:
            return self._board_2_config()
        if self == EdgeTileSettings.BOARD_3:
            return self._board_3_config()
        if self == EdgeTileSettings.BOARD_4:
            return self._board_4_config()

        msg = f"Unknown edge tile setting: {self}"
        raise ValueError(msg)

    def _board_1_config(self) -> dict[tuple[int, int], PatchTile]:
        return {
            (0, 0): PatchTile(color=Color.PINK, pattern=Pattern.VINES),
            (0, 1): PatchTile(color=Color.GREEN, pattern=Pattern.FLOWERS),
            (0, 2): PatchTile(color=Color.BLUE, pattern=Pattern.QUATREFOIL),
            (0, 3): PatchTile(color=Color.NAVY, pattern=Pattern.FERNS),
            (0, 4): PatchTile(color=Color.PURPLE, pattern=Pattern.STRIPES),
            (0, 5): PatchTile(color=Color.YELLOW, pattern=Pattern.DOTS),
            (0, 6): PatchTile(color=Color.PINK, pattern=Pattern.FLOWERS),
            (1, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.STRIPES),
            (2, 0): PatchTile(color=Color.NAVY, pattern=Pattern.VINES),
            (3, 0): PatchTile(color=Color.PINK, pattern=Pattern.FERNS),
            (4, 0): PatchTile(color=Color.PURPLE, pattern=Pattern.QUATREFOIL),
            (5, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.FLOWERS),
            (6, 0): PatchTile(color=Color.GREEN, pattern=Pattern.STRIPES),
            (6, 1): PatchTile(color=Color.BLUE, pattern=Pattern.DOTS),
            (6, 2): PatchTile(color=Color.PURPLE, pattern=Pattern.VINES),
            (6, 3): PatchTile(color=Color.YELLOW, pattern=Pattern.FERNS),
            (6, 4): PatchTile(color=Color.GREEN, pattern=Pattern.QUATREFOIL),
            (6, 5): PatchTile(color=Color.BLUE, pattern=Pattern.FLOWERS),
            (6, 6): PatchTile(color=Color.NAVY, pattern=Pattern.STRIPES),
            (1, 6): PatchTile(color=Color.NAVY, pattern=Pattern.QUATREFOIL),
            (2, 6): PatchTile(color=Color.PINK, pattern=Pattern.FLOWERS),
            (3, 6): PatchTile(color=Color.BLUE, pattern=Pattern.FERNS),
            (4, 6): PatchTile(color=Color.GREEN, pattern=Pattern.VINES),
            (5, 6): PatchTile(color=Color.PINK, pattern=Pattern.DOTS),
        }

    def _board_2_config(self) -> dict[tuple[int, int], PatchTile]:
        return {
            (0, 0): PatchTile(color=Color.BLUE, pattern=Pattern.FERNS),
            (0, 1): PatchTile(color=Color.PURPLE, pattern=Pattern.DOTS),
            (0, 2): PatchTile(color=Color.PINK, pattern=Pattern.QUATREFOIL),
            (0, 3): PatchTile(color=Color.NAVY, pattern=Pattern.STRIPES),
            (0, 4): PatchTile(color=Color.YELLOW, pattern=Pattern.FLOWERS),
            (0, 5): PatchTile(color=Color.PURPLE, pattern=Pattern.VINES),
            (0, 6): PatchTile(color=Color.GREEN, pattern=Pattern.STRIPES),
            (1, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.VINES),
            (2, 0): PatchTile(color=Color.GREEN, pattern=Pattern.FERNS),
            (3, 0): PatchTile(color=Color.BLUE, pattern=Pattern.STRIPES),
            (4, 0): PatchTile(color=Color.PURPLE, pattern=Pattern.QUATREFOIL),
            (5, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.DOTS),
            (6, 0): PatchTile(color=Color.NAVY, pattern=Pattern.VINES),
            (6, 1): PatchTile(color=Color.PINK, pattern=Pattern.FLOWERS),
            (6, 2): PatchTile(color=Color.PURPLE, pattern=Pattern.FERNS),
            (6, 3): PatchTile(color=Color.YELLOW, pattern=Pattern.STRIPES),
            (6, 4): PatchTile(color=Color.NAVY, pattern=Pattern.QUATREFOIL),
            (6, 5): PatchTile(color=Color.PINK, pattern=Pattern.DOTS),
            (6, 6): PatchTile(color=Color.GREEN, pattern=Pattern.VINES),
            (1, 6): PatchTile(color=Color.BLUE, pattern=Pattern.DOTS),
            (2, 6): PatchTile(color=Color.GREEN, pattern=Pattern.QUATREFOIL),
            (3, 6): PatchTile(color=Color.PINK, pattern=Pattern.STRIPES),
            (4, 6): PatchTile(color=Color.NAVY, pattern=Pattern.FERNS),
            (5, 6): PatchTile(color=Color.BLUE, pattern=Pattern.FLOWERS),
        }

    def _board_3_config(self) -> dict[tuple[int, int], PatchTile]:
        return {
            (0, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.FLOWERS),
            (0, 1): PatchTile(color=Color.NAVY, pattern=Pattern.QUATREFOIL),
            (0, 2): PatchTile(color=Color.BLUE, pattern=Pattern.STRIPES),
            (0, 3): PatchTile(color=Color.GREEN, pattern=Pattern.FERNS),
            (0, 4): PatchTile(color=Color.PINK, pattern=Pattern.VINES),
            (0, 5): PatchTile(color=Color.NAVY, pattern=Pattern.DOTS),
            (0, 6): PatchTile(color=Color.PURPLE, pattern=Pattern.FERNS),
            (1, 0): PatchTile(color=Color.PINK, pattern=Pattern.DOTS),
            (2, 0): PatchTile(color=Color.PURPLE, pattern=Pattern.FLOWERS),
            (3, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.FERNS),
            (4, 0): PatchTile(color=Color.NAVY, pattern=Pattern.STRIPES),
            (5, 0): PatchTile(color=Color.PINK, pattern=Pattern.QUATREFOIL),
            (6, 0): PatchTile(color=Color.GREEN, pattern=Pattern.DOTS),
            (6, 1): PatchTile(color=Color.BLUE, pattern=Pattern.VINES),
            (6, 2): PatchTile(color=Color.NAVY, pattern=Pattern.FLOWERS),
            (6, 3): PatchTile(color=Color.PINK, pattern=Pattern.FERNS),
            (6, 4): PatchTile(color=Color.GREEN, pattern=Pattern.STRIPES),
            (6, 5): PatchTile(color=Color.BLUE, pattern=Pattern.QUATREFOIL),
            (6, 6): PatchTile(color=Color.PURPLE, pattern=Pattern.DOTS),
            (1, 6): PatchTile(color=Color.YELLOW, pattern=Pattern.QUATREFOIL),
            (2, 6): PatchTile(color=Color.PURPLE, pattern=Pattern.STRIPES),
            (3, 6): PatchTile(color=Color.BLUE, pattern=Pattern.FERNS),
            (4, 6): PatchTile(color=Color.GREEN, pattern=Pattern.FLOWERS),
            (5, 6): PatchTile(color=Color.YELLOW, pattern=Pattern.VINES),
        }

    def _board_4_config(self) -> dict[tuple[int, int], PatchTile]:
        return {
            (0, 0): PatchTile(color=Color.GREEN, pattern=Pattern.QUATREFOIL),
            (0, 1): PatchTile(color=Color.PINK, pattern=Pattern.FERNS),
            (0, 2): PatchTile(color=Color.PURPLE, pattern=Pattern.VINES),
            (0, 3): PatchTile(color=Color.YELLOW, pattern=Pattern.DOTS),
            (0, 4): PatchTile(color=Color.GREEN, pattern=Pattern.FLOWERS),
            (0, 5): PatchTile(color=Color.BLUE, pattern=Pattern.QUATREFOIL),
            (0, 6): PatchTile(color=Color.NAVY, pattern=Pattern.VINES),
            (1, 0): PatchTile(color=Color.BLUE, pattern=Pattern.FLOWERS),
            (2, 0): PatchTile(color=Color.YELLOW, pattern=Pattern.STRIPES),
            (3, 0): PatchTile(color=Color.PURPLE, pattern=Pattern.DOTS),
            (4, 0): PatchTile(color=Color.BLUE, pattern=Pattern.VINES),
            (5, 0): PatchTile(color=Color.GREEN, pattern=Pattern.FERNS),
            (6, 0): PatchTile(color=Color.PINK, pattern=Pattern.QUATREFOIL),
            (6, 1): PatchTile(color=Color.NAVY, pattern=Pattern.FLOWERS),
            (6, 2): PatchTile(color=Color.BLUE, pattern=Pattern.STRIPES),
            (6, 3): PatchTile(color=Color.GREEN, pattern=Pattern.DOTS),
            (6, 4): PatchTile(color=Color.PINK, pattern=Pattern.VINES),
            (6, 5): PatchTile(color=Color.NAVY, pattern=Pattern.FERNS),
            (6, 6): PatchTile(color=Color.YELLOW, pattern=Pattern.QUATREFOIL),
            (1, 6): PatchTile(color=Color.PURPLE, pattern=Pattern.FERNS),
            (2, 6): PatchTile(color=Color.YELLOW, pattern=Pattern.VINES),
            (3, 6): PatchTile(color=Color.NAVY, pattern=Pattern.DOTS),
            (4, 6): PatchTile(color=Color.PINK, pattern=Pattern.STRIPES),
            (5, 6): PatchTile(color=Color.PURPLE, pattern=Pattern.FLOWERS),
        }
