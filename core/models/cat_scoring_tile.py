from __future__ import annotations

import itertools
from enum import Enum

from pydantic import BaseModel, Field

from core.models.quilt_board import HexPosition, QuiltBoard


class CatDifficultyGroup(int, Enum):
    """Dot difficulty groups used during setup selection."""

    ONE_DOT = 1
    TWO_DOT = 2
    THREE_DOT = 3


class CatShapeType(str, Enum):
    """Known shape-based requirements in Calico (see docs/CAT_SCORING_TILES.md)."""

    TRIANGLE_3 = "triangle_3"
    LINE_3 = "line_3"
    LINE_4 = "line_4"
    T_SHAPE_5 = "t_shape_5"
    LINE_5 = "line_5"


class GroupSizeRequirement(BaseModel):
    """Size-based requirement: contiguous group of one pattern with size >= n."""

    min_size: int = Field(..., ge=3, description="Minimum contiguous group size (>=3)")


class ShapeRequirement(BaseModel):
    """Shape-based requirement: exact arrangement in one pattern.

    The specific shape encoding is abstracted as an enum here.
    """

    shape: CatShapeType


class CatScoringTile(BaseModel):
    """A cat scoring tile with requirement and two allowed patterns.

    - name: the cat tile name (e.g., Millie, Rumi)
    - difficulty_group: one of the three dot groups (1, 2, or 3 dots)
    - requirement: either a GroupSizeRequirement or ShapeRequirement
    - token_values_desc: ordered high-to-low stack of token values for this cat
    """

    name: str
    difficulty_group: CatDifficultyGroup
    requirement: GroupSizeRequirement | ShapeRequirement
    token_values_desc: list[int] = Field(
        default_factory=list,
        description="Highest to lowest remaining values for this cat's tokens.",
    )

    def get_leo_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_straight_line_patches(quilt_board, 5)

    def get_rumi_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_straight_line_patches(quilt_board, 3)

    def get_tecolote_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_straight_line_patches(quilt_board, 4)

    def _get_straight_line_patches(
        self, quilt_board: QuiltBoard | None = None, length: int = 5
    ) -> list[list[HexPosition]]:
        """Get all possible sets of patch tiles in a straight line.

        Searches for tiles in a straight line where:
        - Starting from any tile on the board (including edge tiles, excluding design goal tiles)
        - Travel length-1 more tiles in each of the 6 directions
        - All tiles must be on the board
        - None of the tiles are design goal tiles
        - Multiple edge tiles allowed if they all have the same pattern (requires quilt_board parameter)
        - If no quilt_board provided, falls back to at most 1 edge tile rule

        Args:
            quilt_board: Optional QuiltBoard instance to check edge tile patterns
            length: Number of tiles to include in the straight line (default: 5)

        Returns:
            List of valid tile combinations, each as a list of HexPosition objects
        """
        valid_combinations = []

        # Iterate through all positions on the 7x7 board
        for q in range(7):
            for r in range(7):
                start_pos = HexPosition(q=q, r=r)

                # Skip design goal positions as starting points
                if start_pos.is_design_goal_position():
                    continue

                # Try each of the 6 directions
                for direction_index in range(6):
                    line_positions = [start_pos]
                    current_pos = start_pos

                    # Travel 4 more steps in this direction
                    valid_line = True
                    for _ in range(length - 1):
                        neighbors = current_pos.get_neighbors(filtered=False)

                        next_pos = neighbors[direction_index]

                        # Check if next position is still on the board
                        if (not next_pos.is_valid) or next_pos.is_design_goal_position():
                            valid_line = False
                            break

                        line_positions.append(next_pos)
                        current_pos = next_pos

                    # If we couldn't build a complete line of 5, skip
                    if not valid_line or len(line_positions) != length:
                        continue

                    # Check edge tile constraints
                    edge_positions = [pos for pos in line_positions if pos.is_edge_position()]

                    edge_patterns = [quilt_board.tiles_by_pos[pos].pattern for pos in edge_positions]

                    if len(set(edge_patterns)) > 1:
                        continue

                    # Sort positions for consistent ordering
                    sorted_line = sorted(line_positions, key=lambda p: p.abs)

                    # Avoid duplicates (lines can be found from both ends)
                    if sorted_line not in valid_combinations:
                        valid_combinations.append(sorted_line)

        return valid_combinations

    def get_callie_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self.get_callie_patches(quilt_board, all_neighbors=True)

    # def get_millie_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
    #     return self.get_callie_patches(quilt_board, all_neighbors=False)

    def _get_callie_patches(
        self, quilt_board: QuiltBoard | None = None, all_neighbors: bool = False
    ) -> list[list[HexPosition]]:
        """Get all possible sets of 3 patch tiles in a triangular arrangement for the Callie cat.

        Searches for 3 tiles in a triangular arrangement where:
        - Starting from any tile on the board (including edge tiles, excluding design goal tiles)
        - Find neighboring tiles that form a triangular pattern
        - All tiles must be on the board
        - None of the tiles are design goal tiles
        - Multiple edge tiles allowed if they all have the same pattern (requires quilt_board parameter)
        - If no quilt_board provided, falls back to at most 1 edge tile rule

        Args:
            quilt_board: Optional QuiltBoard instance to check edge tile patterns
            all_neighbors: Whether to require all three tiles to be mutual neighbors (default: False)

        Returns:
            List of valid 3-tile combinations, each as a list of HexPosition objects
        """
        valid_combinations = []

        # Iterate through all positions on the 7x7 board
        for q, r in itertools.product(range(7), range(7)):
            start_pos = HexPosition(q=q, r=r)

            # Skip design goal positions as starting points
            if start_pos.is_design_goal_position():
                continue

            neighbors = start_pos.get_neighbors(filtered=False)
            for neighbor in neighbors:
                if neighbor.is_design_goal_position() or (not neighbor.is_valid):
                    continue

                next_neighbors = neighbor.get_neighbors(filtered=False)
                for next_neighbor in next_neighbors:
                    if next_neighbor.is_design_goal_position() or (not next_neighbor.is_valid):
                        continue
                    if next_neighbor == start_pos:
                        continue
                    if next_neighbor not in neighbors and all_neighbors:
                        continue

                    line_positions = [start_pos, neighbor, next_neighbor]
                    edge_positions = [pos for pos in line_positions if pos.is_edge_position()]
                    edge_patterns = [quilt_board.tiles_by_pos[pos].pattern for pos in edge_positions]

                    if len(set(edge_patterns)) > 1:
                        continue

                    # Sort positions for consistent ordering
                    sorted_line = sorted(line_positions, key=lambda p: p.abs)

                    # Avoid duplicates (lines can be found from both ends)
                    if sorted_line not in valid_combinations:
                        valid_combinations.append(sorted_line)

        return valid_combinations

    def _build_set(
        self,
        position: HexPosition,
        quilt_board: QuiltBoard | None = None,
        length: int = 3,
        combination: list[HexPosition] | None = None,
    ) -> list[list[HexPosition]]:
        if combination is None:
            combination = []

        valid_combinations = []
        neighbors = position.get_neighbors(filtered=False)
        for neighbor in neighbors:
            if neighbor.is_design_goal_position() or (not neighbor.is_valid) or neighbor == position:
                continue

            if neighbor in combination:
                continue

            edge_positions = [pos for pos in combination + [neighbor] if pos.is_edge_position()]
            edge_patterns = [quilt_board.tiles_by_pos[pos].pattern for pos in edge_positions]
            if len(set(edge_patterns)) > 1:
                continue

            if len(combination) == length - 1:
                valid_combinations.append(combination + [neighbor])
                continue

            result = self._build_set(neighbor, quilt_board, length, combination + [neighbor])
            candidates = [sorted(r, key=lambda p: p.abs) for r in result]
            filtered_candidates = [c for c in candidates if c not in valid_combinations]
            valid_combinations.extend(filtered_candidates)
        return valid_combinations

    def get_almond_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        """Get all possible sets of 3 patch tiles in a straight line for the Callie cat.

        Searches for 5 tiles in a straight line where:
        - Starting from any tile on the board (including edge tiles, excluding design goal tiles)
        - Travel 4 more tiles in each of the 6 directions (northwest, northeast, east, west, southeast, southwest)
        - All 5 tiles must be on the board
        - None of the 5 tiles are design goal tiles
        - Multiple edge tiles allowed if they all have the same pattern (requires quilt_board parameter)
        - If no quilt_board provided, falls back to at most 1 edge tile rule

        Args:
            quilt_board: Optional QuiltBoard instance to check edge tile patterns

        Returns:
            List of valid 5-tile combinations, each as a list of HexPosition objects
        """
        valid_combinations = []

        # Iterate through all positions on the 7x7 board
        for q, r in itertools.product(range(7), range(7)):
            start_pos = HexPosition(q=q, r=r)

            # Skip design goal positions as starting points
            if start_pos.is_design_goal_position():
                continue

            candidates = self._build_set(start_pos, quilt_board, 5)
            for candidate in candidates:
                if not candidate[0].is_neighbor(candidate[1]):
                    continue
                if not candidate[0].is_neighbor(candidate[2]):
                    continue
                if not candidate[0].is_neighbor(candidate[3]):
                    continue
                if candidate[0].is_neighbor(candidate[4]):
                    continue
                if not candidate[1].is_neighbor(candidate[3]):
                    continue
                if not candidate[1].is_neighbor(candidate[4]):
                    continue
                if candidate[1].is_neighbor(candidate[2]):
                    continue
                if not candidate[2].is_neighbor(candidate[3]):
                    continue
                if candidate[2].is_neighbor(candidate[4]):
                    continue
                if not candidate[3].is_neighbor(candidate[4]):
                    continue
                sorted_line = sorted(candidate, key=lambda p: p.abs)

                # Avoid duplicates (lines can be found from both ends)
                if sorted_line not in valid_combinations:
                    valid_combinations.append(sorted_line)

        return valid_combinations

    def get_millie_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_patches_by_length(quilt_board, 3)

    def get_tibbit_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_patches_by_length(quilt_board, 4)

    def get_coconuts_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_patches_by_length(quilt_board, 5)

    def get_cira_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_patches_by_length(quilt_board, 6)

    def get_gwenivere_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        return self._get_patches_by_length(quilt_board, 7)

    def _get_patches_by_length(self, quilt_board: QuiltBoard | None = None, length: int = 3) -> list[list[HexPosition]]:
        """Get all possible sets of patch tiles in connected groups of specified length.

        Searches for connected groups of tiles where:
        - Starting from any tile on the board (including edge tiles, excluding design goal tiles)
        - Find all connected groups of the specified length
        - All tiles must be on the board
        - None of the tiles are design goal tiles
        - Multiple edge tiles allowed if they all have the same pattern (requires quilt_board parameter)
        - If no quilt_board provided, falls back to at most 1 edge tile rule

        Args:
            quilt_board: Optional QuiltBoard instance to check edge tile patterns
            length: Number of tiles to include in the connected group (default: 3)

        Returns:
            List of valid tile combinations, each as a list of HexPosition objects
        """
        valid_combinations = []

        # Iterate through all positions on the 7x7 board
        for q, r in itertools.product(range(7), range(7)):
            start_pos = HexPosition(q=q, r=r)

            # Skip design goal positions as starting points
            if start_pos.is_design_goal_position():
                continue

            candidates = self._build_set(start_pos, quilt_board, length)
            for candidate in candidates:
                sorted_line = sorted(candidate, key=lambda p: p.abs)

                # Avoid duplicates (lines can be found from both ends)
                if sorted_line not in valid_combinations:
                    valid_combinations.append(sorted_line)

        return valid_combinations
