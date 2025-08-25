from __future__ import annotations

import itertools
import json
from enum import Enum
from pathlib import Path

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
        patches = self._get_straight_line_patches(quilt_board, 5)
        self.save_patches(patches)
        return patches

    def get_rumi_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_straight_line_patches(quilt_board, 3)
        self.save_patches(patches)
        return patches

    def get_tecolote_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_straight_line_patches(quilt_board, 4)
        self.save_patches(patches)
        return patches

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
        patches = self._get_callie_patches(quilt_board, all_neighbors=True)
        self.save_patches(patches)
        return patches

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

        almond_rorations = [
            [0, 1, 2, 3, 4],
            [1, 3, 0, 2, 4],
            [2, 4, 0, 1, 3],
            [4, 3, 2, 1, 0],
            [3, 1, 4, 2, 0],
            [2, 0, 4, 3, 1],
        ]

        # Iterate through all positions on the 7x7 board
        for q, r in itertools.product(range(7), range(7)):
            start_pos = HexPosition(q=q, r=r)

            # Skip design goal positions as starting points
            if start_pos.is_design_goal_position():
                continue

            candidates = self._build_set(start_pos, quilt_board, 5)
            for candidate in candidates:
                for rotation in almond_rorations:
                    rotated_candidate = [candidate[i] for i in rotation]
                    if not rotated_candidate[0].is_neighbor(rotated_candidate[1]):
                        continue
                    if not rotated_candidate[0].is_neighbor(rotated_candidate[2]):
                        continue
                    if not rotated_candidate[0].is_neighbor(rotated_candidate[3]):
                        continue
                    if rotated_candidate[0].is_neighbor(rotated_candidate[4]):
                        continue
                    if not rotated_candidate[1].is_neighbor(rotated_candidate[3]):
                        continue
                    if not rotated_candidate[1].is_neighbor(rotated_candidate[4]):
                        continue
                    if rotated_candidate[1].is_neighbor(rotated_candidate[2]):
                        continue
                    if not rotated_candidate[2].is_neighbor(rotated_candidate[3]):
                        continue
                    if rotated_candidate[2].is_neighbor(rotated_candidate[4]):
                        continue
                    if not rotated_candidate[3].is_neighbor(rotated_candidate[4]):
                        continue
                    sorted_line = sorted(rotated_candidate, key=lambda p: p.abs)

                    # Avoid duplicates (lines can be found from both ends)
                    if sorted_line not in valid_combinations:
                        valid_combinations.append(sorted_line)
                        break

        self.save_patches(valid_combinations)
        return valid_combinations

    def get_millie_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_patches_by_length(quilt_board, 3)
        self.save_patches(patches)
        return patches

    def get_tibbit_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_patches_by_length(quilt_board, 4)
        self.save_patches(patches)
        return patches

    def get_coconut_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_patches_by_length(quilt_board, 5)
        self.save_patches(patches)
        return patches

    def get_cira_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_patches_by_length(quilt_board, 6)
        self.save_patches(patches)
        return patches

    def get_gwenivere_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        patches = self._get_patches_by_length(quilt_board, 7)
        self.save_patches(patches)
        return patches

    def get_patches(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        if self.name == "Callie":
            return self.get_callie_patches(quilt_board)
        if self.name == "Millie":
            return self.get_millie_patches(quilt_board)
        if self.name == "Tibbit":
            return self.get_tibbit_patches(quilt_board)
        if self.name == "Coconut":
            return self.get_coconut_patches(quilt_board)
        if self.name == "Cira":
            return self.get_cira_patches(quilt_board)
        if self.name == "Gwenivere":
            return self.get_gwenivere_patches(quilt_board)
        if self.name == "Almond":
            return self.get_almond_patches(quilt_board)
        if self.name == "Leo":
            return self.get_leo_patches(quilt_board)
        if self.name == "Rumi":
            return self.get_rumi_patches(quilt_board)
        if self.name == "Tecolote":
            return self.get_tecolote_patches(quilt_board)
        raise ValueError(f"No get_patches method found for cat: {self.name}")

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

    def save_patches(self, patches: list[list[HexPosition]], data_dir: str = "data") -> None:
        """Save the patches data for this cat to a JSON file.

        Creates a data directory if it doesn't exist and saves the patches as a JSON file
        named after the cat. Each patch is converted from HexPosition objects to abs values.

        Args:
            patches: List of lists of HexPosition objects to save
            data_dir: Directory name to save the data files (default: "data")
        """
        # Create data directory if it doesn't exist
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)

        # Convert HexPosition objects to abs values
        abs_patches = [[pos.abs for pos in patch] for patch in patches]

        # Sort patches by their abs values for consistent ordering
        abs_patches.sort()

        # Save to JSON file named after the cat
        filename = f"{self.name.lower()}_patches.json"
        filepath = data_path / filename

        with open(filepath, "w") as f:
            f.write("[\n")
            for i, patch in enumerate(abs_patches):
                if i > 0:
                    f.write(",\n")
                f.write(f"  {json.dumps(patch)}")
            f.write("\n]")

    def load_patches(self, data_dir: str = "data") -> list[list[int]]:
        """Load the patches data for this cat from a JSON file.

        Args:
            data_dir: Directory name where the data files are stored (default: "data")

        Returns:
            List of lists containing abs values for each patch

        Raises:
            FileNotFoundError: If the JSON file for this cat doesn't exist
        """
        data_path = Path(data_dir)
        filename = f"{self.name.lower()}_patches.json"
        filepath = data_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Patches data file not found: {filepath}")

        with open(filepath) as f:
            return json.load(f)

    def _get_patches_for_cat(self, quilt_board: QuiltBoard | None = None) -> list[list[HexPosition]]:
        """Get patches for this specific cat by calling the appropriate method.

        Args:
            quilt_board: Optional QuiltBoard instance to pass to the get_patches method

        Returns:
            List of valid tile combinations for this cat

        Raises:
            ValueError: If no matching get_patches method is found for this cat
        """
        cat_name_lower = self.name.lower()
        method_name = f"get_{cat_name_lower}_patches"

        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(quilt_board)
        raise ValueError(f"No get_patches method found for cat: {self.name}")
