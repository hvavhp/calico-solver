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

    RAINBOW_STRIPES = "rainbow_stripes"
    """Edge tiles with all 6 colors in alternating stripe patterns."""

    WARM_COOL_SPLIT = "warm_cool_split"
    """Top/bottom edges use warm colors, left/right edges use cool colors."""

    PATTERN_FOCUS = "pattern_focus"
    """Edge tiles emphasize pattern diversity with balanced color distribution."""

    SYMMETRICAL = "symmetrical"
    """Edge tiles arranged in perfect rotational symmetry around the board."""

    def get_edge_tiles(self) -> dict[tuple[int, int], PatchTile]:
        """Returns the edge tile configuration as position -> PatchTile mapping.

        Uses (q, r) axial coordinates where the 7x7 board spans from (0,0) to (6,6).
        Edge positions are those on the border of this coordinate space.
        """
        if self == EdgeTileSettings.RAINBOW_STRIPES:
            return self._rainbow_stripes_config()
        if self == EdgeTileSettings.WARM_COOL_SPLIT:
            return self._warm_cool_split_config()
        if self == EdgeTileSettings.PATTERN_FOCUS:
            return self._pattern_focus_config()
        if self == EdgeTileSettings.SYMMETRICAL:
            return self._symmetrical_config()
        
        msg = f"Unknown edge tile setting: {self}"
        raise ValueError(msg)

    def _rainbow_stripes_config(self) -> dict[tuple[int, int], PatchTile]:
        """Creates rainbow edge with alternating stripe patterns."""
        edge_tiles = {}
        colors = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.PURPLE, Color.PINK]
        patterns = [Pattern.STRIPES, Pattern.POLKA_DOTS]

        # Get all edge positions for a 7x7 hex board
        edge_positions = self._get_edge_positions()

        for i, pos in enumerate(edge_positions):
            color = colors[i % 6]
            pattern = patterns[i % 2]
            edge_tiles[pos] = PatchTile(color=color, pattern=pattern)

        return edge_tiles

    def _warm_cool_split_config(self) -> dict[tuple[int, int], PatchTile]:
        """Creates warm colors on top/bottom, cool on sides."""
        edge_tiles = {}
        warm_colors = [Color.YELLOW, Color.ORANGE, Color.PINK]
        cool_colors = [Color.BLUE, Color.GREEN, Color.PURPLE]
        patterns = [Pattern.POLKA_DOTS, Pattern.PLAID, Pattern.STRIPES]

        edge_positions = self._get_edge_positions()

        for i, pos in enumerate(edge_positions):
            q, r = pos
            # Top and bottom edges get warm colors
            if r == 0 or r == 6:
                color = warm_colors[i % 3]
            # Left and right edges get cool colors
            else:
                color = cool_colors[i % 3]
            pattern = patterns[i % 3]
            edge_tiles[pos] = PatchTile(color=color, pattern=pattern)

        return edge_tiles

    def _pattern_focus_config(self) -> dict[tuple[int, int], PatchTile]:
        """Emphasizes pattern diversity with balanced colors."""
        edge_tiles = {}
        colors = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.PURPLE, Color.PINK]
        patterns = [
            Pattern.POLKA_DOTS,
            Pattern.STRIPES,
            Pattern.PLAID,
            Pattern.FLORAL,
            Pattern.VINES,
            Pattern.HERRINGBONE,
        ]

        edge_positions = self._get_edge_positions()

        for i, pos in enumerate(edge_positions):
            color = colors[i % 6]
            pattern = patterns[i % 6]
            edge_tiles[pos] = PatchTile(color=color, pattern=pattern)

        return edge_tiles

    def _symmetrical_config(self) -> dict[tuple[int, int], PatchTile]:
        """Creates rotational symmetry around the board center."""
        edge_tiles = {}
        colors = [Color.BLUE, Color.YELLOW, Color.PURPLE]
        patterns = [Pattern.POLKA_DOTS, Pattern.STRIPES, Pattern.HERRINGBONE]

        edge_positions = self._get_edge_positions()

        for i, pos in enumerate(edge_positions):
            # Create 3-fold symmetry pattern
            color = colors[i % 3]
            pattern = patterns[(i // 8) % 3]  # Change pattern every 8 positions
            edge_tiles[pos] = PatchTile(color=color, pattern=pattern)

        return edge_tiles

    def _get_edge_positions(self) -> list[tuple[int, int]]:
        """Returns all edge positions for a 7x7 hex board in axial coordinates.

        Edge positions are those where q=0, q=6, r=0, r=6, or q+r=0, q+r=6.
        """
        edge_positions = []

        # Collect all edge positions
        for q in range(7):
            for r in range(7):
                # Check if position is on the edge of a 7x7 hex grid
                if q == 0 or q == 6 or r == 0 or r == 6 or q + r == 0 or q + r == 6 or q + r == 12:
                    # Additional boundary check for hex shape
                    if 0 <= q <= 6 and 0 <= r <= 6 and 0 <= q + r <= 12:
                        edge_positions.append((q, r))

        # Sort for consistent ordering
        edge_positions.sort()
        return edge_positions
