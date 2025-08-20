#!/usr/bin/env python3
"""Test program to print out all two neighbor tile sets near edge tiles."""

from core.enums.edge_tile_settings import EdgeTileSettings
from core.models.design_goal_tile import DesignGoalTile
from core.models.quilt_board import HexPosition, QuiltBoard


def main():
    """Print all two neighbor tile sets near edge tiles."""
    # Create dummy design goal tiles
    goal_tiles = [
        DesignGoalTile(config="A", lower_points=3, higher_points=5),
        DesignGoalTile(config="B", lower_points=4, higher_points=6),
        DesignGoalTile(config="C", lower_points=2, higher_points=4),
    ]

    # Create a QuiltBoard with edge settings and design goal tiles
    board = QuiltBoard(edge_setting=EdgeTileSettings.BOARD_1, design_goal_tiles=goal_tiles)

    # Get all two neighbor tile sets near edge
    tile_pairs = board.get_two_neighbor_tile_sets_near_edge()

    print(f"Found {len(tile_pairs)} pairs of neighboring tiles near edge tiles:")
    print()

    for i, tile_pair in enumerate(tile_pairs, 1):
        print(f"Pair {i:2d}: ({tile_pair[0].q},{tile_pair[0].r}) - ({tile_pair[1].q},{tile_pair[1].r})")

    print()
    print("Note: These are pairs where at least one tile is adjacent to an edge tile.")
    print("Edge tiles are at the border positions of the 7x7 grid.")

    # Also show which tiles are near edges for reference
    edge_tiles = []
    tiles_near_edge = set()

    for q in range(7):
        for r in range(7):
            pos = HexPosition(q=q, r=r)
            if board._is_edge_position(pos):
                edge_tiles.append(pos)

    for edge_tile in edge_tiles:
        neighbors = board._get_hex_neighbors(edge_tile)
        for neighbor in neighbors:
            if not board._is_edge_position(neighbor):
                tiles_near_edge.add(neighbor)

    print(f"\nTiles adjacent to edge tiles: {sorted([(t.q, t.r) for t in tiles_near_edge])}")


if __name__ == "__main__":
    main()
