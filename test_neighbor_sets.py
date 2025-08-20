#!/usr/bin/env python3
"""Test program to print out all three neighbor tile sets using the QuiltBoard method."""

from core.enums.edge_tile_settings import EdgeTileSettings
from core.models.design_goal_tile import DesignGoalTile
from core.models.quilt_board import HexPosition, QuiltBoard


def main():
    """Print all three neighbor tile sets from a QuiltBoard."""
    # Create dummy design goal tiles
    goal_tiles = [
        DesignGoalTile(config="A", lower_points=3, higher_points=5),
        DesignGoalTile(config="B", lower_points=4, higher_points=6),
        DesignGoalTile(config="C", lower_points=2, higher_points=4),
    ]

    # Create a QuiltBoard with edge settings and design goal tiles
    board = QuiltBoard(edge_setting=EdgeTileSettings.BOARD_1, design_goal_tiles=goal_tiles)

    # Get all three neighbor tile sets
    tile_sets = board.get_three_neighbor_tile_sets()

    print(f"Found {len(tile_sets)} sets of three neighboring tiles:")
    print()

    for i, tile_set in enumerate(tile_sets, 1):
        print(f"Set {i:2d}: ", end="")
        for j, pos in enumerate(tile_set):
            if j > 0:
                print(" - ", end="")
            print(f"({pos.q},{pos.r})", end="")
        print()

    print()
    print("Note: Edge tiles and design goal tiles are excluded.")
    print("Design goal tiles are at positions: (4,3), (5,4), (3,5)")
    print("Edge tiles are at the border positions of the 7x7 grid.")

    # Also print the valid positions for reference
    print("\nValid positions (non-edge, non-design-goal):")
    valid_positions = []
    for q in range(7):
        for r in range(7):
            pos = HexPosition(q=q, r=r)
            if not board._is_edge_position(pos) and not board._is_design_goal_position(pos):
                valid_positions.append(pos)

    print("Valid positions:", [f"({pos.q},{pos.r})" for pos in valid_positions])


if __name__ == "__main__":
    main()
