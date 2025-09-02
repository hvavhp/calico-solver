import json
import time
from itertools import combinations

import numpy as np
from ortools.sat.python import cp_model

from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import ALL_PATTERNS, PATTERN_MAP, Pattern
from core.models.cat_scoring_tile import CatDifficultyGroup
from core.models.quilt_board import HexPosition, QuiltBoard
from solvers.restructured_design_goals_solver import DesignGoalsModel
from solvers.restructured_design_goals_solver import build_model as design_goals_build_model

# Available cats from data folder
AVAILABLE_CATS = ["almond", "cira", "coconut", "gwenivere", "leo", "millie", "rumi", "tecolote", "tibbit"]

# Map cat names to their difficulty groups
CAT_DIFFICULTY_GROUPS = {
    "millie": CatDifficultyGroup.ONE_DOT,
    "tibbit": CatDifficultyGroup.ONE_DOT,
    "callie": CatDifficultyGroup.ONE_DOT,
    "rumi": CatDifficultyGroup.ONE_DOT,
    "coconut": CatDifficultyGroup.TWO_DOT,
    "cira": CatDifficultyGroup.TWO_DOT,
    "tecolote": CatDifficultyGroup.TWO_DOT,
    "almond": CatDifficultyGroup.TWO_DOT,
    "gwenivere": CatDifficultyGroup.THREE_DOT,
    "leo": CatDifficultyGroup.THREE_DOT,
}

# Board configuration - always use board 1
BOARD_CONFIG = EdgeTileSettings.BOARD_1

# Cat weights (you can adjust these based on your preferences)
CAT_WEIGHTS = {
    "leo": 11,
    "cira": 9,
    "rumi": 5,
    "almond": 9,
    "callie": 3,
    "coconut": 7,
    "gwenivere": 11,
    "millie": 3,
    "tecolote": 7,
    "tibbit": 5,
}


def is_valid_cat_combination(cat_names):
    """Check if a combination of cats is valid (not all from the same difficulty group)."""
    if len(cat_names) != 3:
        return False

    difficulty_groups = [CAT_DIFFICULTY_GROUPS[cat] for cat in cat_names]
    unique_groups = set(difficulty_groups)

    # Valid combinations must have cats from at least 2 different difficulty groups
    return len(unique_groups) == 3


def load_cat_patches(cat_names):
    """Load patches for specified cats."""
    all_subsets: list[list[int]] = []
    all_weights: list[int] = []
    cat_for_weight = {}  # Map weight to cat name for disambiguation

    for cat_name in cat_names:
        patch_file = f"data/{cat_name}_patches.json"
        try:
            with open(patch_file) as f:
                patches = json.load(f)

            weight = CAT_WEIGHTS.get(cat_name, 1)
            cat_for_weight[weight] = cat_name  # Track which cat this weight belongs to

            for patch in patches:
                all_subsets.append(patch)
                all_weights.append(weight)

        except FileNotFoundError:
            print(f"Warning: Could not find patches file for {cat_name}")
            continue

    return all_subsets, all_weights, cat_for_weight


def build_graph():
    """Build the base hexagonal graph with vertices 0-48."""
    edges = []
    for vertex in range(49):  # 0 to 48 inclusive
        try:
            hex_pos = HexPosition.from_abs(vertex)
            if hex_pos.is_valid:
                neighbors = hex_pos.get_neighbors(filtered=True)
                for neighbor in neighbors:
                    neighbor_idx = neighbor.abs
                    if neighbor_idx <= 48:  # Only include edges to vertices 0-48
                        # Add edge (ensure we don't duplicate by using vertex < neighbor_idx)
                        if vertex < neighbor_idx:
                            edges.append((vertex, neighbor_idx))
        except Exception:
            continue  # Skip invalid vertices
    return edges


def add_virtual_vertices(vertex_edges, subsets, weights):
    """Add virtual vertices for pattern constraints."""
    dummy_design_goals = [
        DesignGoalTiles.SIX_UNIQUE.value,
        DesignGoalTiles.THREE_PAIRS.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
    ]
    board = QuiltBoard(edge_setting=BOARD_CONFIG, design_goal_tiles=dummy_design_goals)

    hex_positions = [HexPosition.from_abs(i) for i in range(49)]
    is_edge_positions = [pos.is_edge_position() for pos in hex_positions]
    patterns = [None if not pos.is_edge_position() else board.tiles_by_pos[pos].pattern for pos in hex_positions]
    pattern_map = {pattern: i + 49 for i, pattern in enumerate(ALL_PATTERNS)}

    subsets.extend([[i] for i in range(49, 49 + len(ALL_PATTERNS))])
    weights.extend([100000000] * len(ALL_PATTERNS))

    vertex_edges.extend(
        [(i, j) for i in range(49, 49 + len(ALL_PATTERNS)) for j in range(49, 49 + len(ALL_PATTERNS)) if i < j]
    )

    # Build the virtual edges
    for i in range(49):
        if not is_edge_positions[i]:
            continue
        pattern = patterns[i]
        virtual_idx = pattern_map[pattern]
        vertex_edges.append((i, virtual_idx))


def build_model(design_goals_model: DesignGoalsModel, cat_names: list[str]):
    """
    Build model with variables and constraints for given cats.
    Always uses board 1.

    Args:
        design_goals_model: DesignGoalsModel instance with CP-SAT model and pattern variables
        cat_names: List of 3 cat names to use

    Returns:
        tuple: (model, variables, subsets, weights, cat_for_weight)
    """
    # Load patches for the specified cats
    subsets, weights, cat_for_weight = load_cat_patches(cat_names)

    pattern_variables: dict[int, cp_model.IntVar] = design_goals_model.pattern_variables
    model = design_goals_model.model

    # Build graph with vertices 0-48
    vertex_edges = build_graph()

    # Add virtual vertices and edges for pattern constraints
    add_virtual_vertices(vertex_edges, subsets, weights)

    vars_count = 49 + 6
    m = len(subsets)

    # Subset-level adjacency if share any edge across the base graph and same label
    same_label_idxs = {}
    for i, w in enumerate(weights):
        same_label_idxs.setdefault(w, []).append(i)

    for _w, idxs in same_label_idxs.items():
        idxs.extend(same_label_idxs[100000000])
    same_label_idxs.pop(100000000)

    dummy_design_goals = [
        DesignGoalTiles.SIX_UNIQUE.value,
        DesignGoalTiles.THREE_PAIRS.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
    ]
    board = QuiltBoard(edge_setting=BOARD_CONFIG, design_goal_tiles=dummy_design_goals)
    pattern_groups: dict[int, dict[Pattern, list[int]]] = {}
    hex_positions = [HexPosition.from_abs(i) for i in range(49)]
    is_edge_positions = [pos.is_edge_position() for pos in hex_positions]
    patterns = [None if not pos.is_edge_position() else board.tiles_by_pos[pos].pattern for pos in hex_positions]

    for w, idxs in same_label_idxs.items():
        groups: dict[Pattern, list[int]] = {}
        for idx in idxs:
            if weights[idx] == 100000000:
                continue
            subset = subsets[idx]
            for v in subset:
                if is_edge_positions[v]:
                    pattern = patterns[v]
                    groups.setdefault(pattern, []).append(idx)
                    break
        for pattern, indices in groups.items():
            pattern_groups.setdefault(w, {}).setdefault(pattern, indices)

    # Bitmasks per subset
    def mask_of(subset):
        msk = 0
        for u in subset:
            msk |= 1 << u
        return msk

    masks = [mask_of(subset) for subset in subsets]

    # Adjacency bitmask on vertices
    nbr = [0] * vars_count
    for u, v in vertex_edges:
        nbr[u] |= 1 << v
        nbr[v] |= 1 << u

    # Per-subset neighbor-vertices mask
    adj_mask = []
    for subset in subsets:
        msk = 0
        for u in subset:
            msk |= nbr[u]
        adj_mask.append(msk)

    def build_adj_numpy_optimized(same_label_idxs, masks, adj_mask, subsets, cat_for_weight, m):
        """NumPy-optimized version of adjacency construction."""
        # Convert to numpy arrays for vectorized operations
        masks_np = np.array(masks, dtype=np.uint64)
        adj_mask_np = np.array(adj_mask, dtype=np.uint64)

        adj = [set() for _ in range(m)]

        for w, idxs in same_label_idxs.items():
            if cat_for_weight.get(w) == "gwenivere":
                continue

            if len(idxs) < 2:
                continue

            # Convert indices to numpy array
            idxs_np = np.array(idxs, dtype=np.int32)
            n_idxs = len(idxs_np)

            # Extract relevant masks and adj_masks
            subset_masks = masks_np[idxs_np]  # Shape: (n_idxs,)
            subset_adj_masks = adj_mask_np[idxs_np]  # Shape: (n_idxs,)

            # Create all pairs using broadcasting
            # i_indices[k, l] = k, j_indices[k, l] = l for all k < l
            i_indices, j_indices = np.meshgrid(np.arange(n_idxs), np.arange(n_idxs), indexing="ij")
            valid_pairs = i_indices < j_indices

            # Get the actual pairs
            i_pair_indices = i_indices[valid_pairs]  # Indices into idxs_np
            j_pair_indices = j_indices[valid_pairs]  # Indices into idxs_np

            # Get masks for these pairs
            i_masks = subset_masks[i_pair_indices]
            j_masks = subset_masks[j_pair_indices]
            i_adj_masks = subset_adj_masks[i_pair_indices]
            j_adj_masks = subset_adj_masks[j_pair_indices]

            # Vectorized adjacency check: (masks[j] & adj_mask[i]) == 0 and (masks[i] & adj_mask[j]) == 0
            no_adj_1 = (j_masks & i_adj_masks) == 0
            no_adj_2 = (i_masks & j_adj_masks) == 0
            not_adjacent = no_adj_1 & no_adj_2

            # Vectorized subset overlap check: masks[i] & masks[j] != 0
            # (equivalent to any(_v in subsets[i] for _v in subsets[j]))
            subset_overlap = (i_masks & j_masks) != 0

            # We add adjacency if NOT (not_adjacent OR subset_overlap)
            # Which is equivalent to: adjacent AND not subset_overlap
            should_add = ~(not_adjacent | subset_overlap)

            # Add the valid adjacencies
            valid_i_indices = i_pair_indices[should_add]
            valid_j_indices = j_pair_indices[should_add]

            for idx_i, idx_j in zip(valid_i_indices, valid_j_indices, strict=False):
                actual_i = idxs_np[idx_i]
                actual_j = idxs_np[idx_j]
                adj[actual_i].add(actual_j)
                adj[actual_j].add(actual_i)

        return adj

    def build_adj_original(same_label_idxs, masks, adj_mask, subsets, cat_for_weight, m):
        """Original adjacency construction for comparison."""
        adj = [set() for _ in range(m)]
        for w, idxs in same_label_idxs.items():
            if cat_for_weight.get(w) == "gwenivere":
                continue
            for a_i in range(len(idxs)):
                i = idxs[a_i]
                for a_j in range(a_i + 1, len(idxs)):
                    j = idxs[a_j]
                    if (masks[j] & adj_mask[i]) == 0 and (masks[i] & adj_mask[j]) == 0:
                        continue
                    if any(_v in subsets[i] for _v in subsets[j]):
                        continue
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # Time both approaches and verify they produce the same result
    print("Building adjacency using original method...")
    start_time = time.time()
    adj_original = build_adj_original(same_label_idxs, masks, adj_mask, subsets, cat_for_weight, m)
    original_time = time.time() - start_time

    print("Building adjacency using NumPy-optimized method...")
    start_time = time.time()
    adj_numpy = build_adj_numpy_optimized(same_label_idxs, masks, adj_mask, subsets, cat_for_weight, m)
    numpy_time = time.time() - start_time

    # Verify both methods produce the same result
    results_match = True
    for i in range(m):
        if adj_original[i] != adj_numpy[i]:
            results_match = False
            print(f"Mismatch at index {i}: original={adj_original[i]}, numpy={adj_numpy[i]}")
            break

    print(f"Original method time: {original_time:.4f}s")
    print(f"NumPy method time: {numpy_time:.4f}s")
    if numpy_time > 0:
        speedup = original_time / numpy_time
        print(f"Speedup: {speedup:.2f}x")
    print(f"Results match: {results_match}")

    # Use the NumPy version if it's correct
    adj = adj_numpy if results_match else adj_original

    # Enumerate forbidden triangles within each same-label partition
    forbidden_triangles = []
    for w, idxs in same_label_idxs.items():
        # Skip building forbidden triangles for gwenivere
        if cat_for_weight.get(w) in ["gwenivere", "cira"]:
            continue

        idxs_sorted = sorted(idxs, key=lambda t: (len(adj[t]), t))
        pos = {u: i for i, u in enumerate(idxs_sorted)}
        # Work with sets as ordered by pos to avoid duplicates
        neigh_sets = {u: {v for v in adj[u] if pos.get(v, -1) > pos[u]} for u in idxs_sorted}
        for i in idxs_sorted:
            neighbors = neigh_sets[i]
            for j in neighbors:
                # Intersection of higher-index neighbors
                common = neighbors & neigh_sets[j]
                for k in common:
                    # These are virtual subsets, so they allow to exist freely
                    if weights[i] == 100000000 and weights[j] == 100000000 and weights[k] == 100000000:
                        continue
                    forbidden_triangles.append((i, j, k))

    outer_pattern_restrictions: list[tuple[int, int]] = []
    for pattern in ALL_PATTERNS:
        groups = [pattern_groups.get(w, {}).get(pattern, []) for w in same_label_idxs]
        for combo in combinations(groups, 2):
            for i in combo[0]:
                s_1 = set(subsets[i])
                for j in combo[1]:
                    s_2 = set(subsets[j])
                    if s_1 & s_2 != set():
                        continue
                    sorted_indices = sorted([i, j])
                    if sorted_indices in outer_pattern_restrictions:
                        continue
                    outer_pattern_restrictions.append(sorted_indices)

    # Create variables
    y = [model.NewBoolVar(f"y[{i}]") for i in range(m)]

    # Add constraints

    # Pattern variable constraints: y[i] can only be 1 if all tiles in subsets[i] have the same pattern
    # More efficient approach: handle edge tiles first, then use first tile as reference for others
    for i in range(m):
        subset = subsets[i]

        # Skip virtual vertices (pattern vertices added in add_virtual_vertices)
        if weights[i] == 100000000:
            continue

        # Only consider subsets with tiles that have pattern variables
        actual_tiles = [tile for tile in subset if tile < 49 and f"P_{tile}" in pattern_variables]

        # Check if any tiles in the subset are edge tiles (have fixed patterns)
        edge_tiles = [tile for tile in subset if is_edge_positions[tile]]

        if edge_tiles:
            # If there are edge tiles, all tiles must match the edge tile's fixed pattern
            edge_tile = edge_tiles[0]  # Use first edge tile as reference
            fixed_pattern = patterns[edge_tile]  # This is the fixed pattern from the board
            reference_tile = actual_tiles[0]
            pattern_value = PATTERN_MAP[fixed_pattern]
            reference_pattern_var = pattern_variables[f"P_{reference_tile}"]
            model.Add(reference_pattern_var == pattern_value).OnlyEnforceIf(y[i])

        # No edge tiles, use first tile as reference for consistency
        reference_tile = actual_tiles[0]
        reference_pattern_var = pattern_variables[f"P_{reference_tile}"]

        # All other tiles must have same pattern as reference when y[i] == 1
        for tile in actual_tiles[1:]:
            tile_pattern_var = pattern_variables[f"P_{tile}"]

            # If y[i] == 1, then tile_pattern_var == reference_pattern_var
            model.Add(tile_pattern_var == reference_pattern_var).OnlyEnforceIf(y[i])

    # Disjointness per vertex
    for v in range(vars_count):
        model.Add(sum(y[i] for i in range(m) if (masks[i] >> v) & 1) <= 1)

    # Triangle cuts
    for i, j, k in forbidden_triangles:
        model.Add(y[i] + y[j] + y[k] <= 2)

    # Make sure that each cat occupy only 2 patterns
    for w, idxs in same_label_idxs.items():
        groups = pattern_groups[w]
        active = []
        for pattern, idxs in groups.items():
            b = model.NewBoolVar(f"b[{w}][{pattern.value}]")
            model.AddMaxEquality(b, [y[idx] for idx in idxs])
            active.append(b)
        model.Add(sum(active) <= 2)

    # Make sure that no two cats occupy the same pattern
    for i, j in outer_pattern_restrictions:
        model.Add(y[i] + y[j] <= 1)

    # # Add pattern difference constraints for neighboring subsets
    # # For each neighboring pair (i, j), create boolean variable indicating if they have different patterns
    # pattern_diff_vars = {}

    # for i in range(m):
    #     if weights[i] == 100000000:  # Skip virtual vertices
    #         continue
    #     for j in adj[i]:
    #         if j <= i or weights[j] == 100000000:  # Avoid duplicates and skip virtual vertices
    #             continue

    #         # Create boolean variable: 1 if patterns are different, 0 if same
    #         diff_var = model.NewBoolVar(f"pattern_diff_{i}_{j}")
    #         pattern_diff_vars[(i, j)] = diff_var

    #         # Get pattern representatives for both subsets
    #         subset_i = subsets[i]
    #         subset_j = subsets[j]

    #         # Find representative tiles (first tiles) that have pattern variables
    #         rep_tile_i = None
    #         rep_tile_j = None

    #         for tile in subset_i:
    #             if tile < 49 and f"P_{tile}" in pattern_variables:
    #                 rep_tile_i = tile
    #                 break

    #         for tile in subset_j:
    #             if tile < 49 and f"P_{tile}" in pattern_variables:
    #                 rep_tile_j = tile
    #                 break

    #         if rep_tile_i is not None and rep_tile_j is not None:
    #             pattern_var_i = pattern_variables[f"P_{rep_tile_i}"]
    #             pattern_var_j = pattern_variables[f"P_{rep_tile_j}"]

    #             # diff_var = 1 if patterns are different, 0 if same
    #             # This is equivalent to: diff_var = (pattern_var_i != pattern_var_j)

    #             # diff_var = 1 iff pattern_var_i != pattern_var_j
    #             model.Add(pattern_var_i != pattern_var_j).OnlyEnforceIf(diff_var)
    #             # model.Add(pattern_var_i == pattern_var_j).OnlyEnforceIf(diff_var.Not())

    # # Add the main constraint: y_i + y_j <= pattern_diff + 1
    # # This means both can only be selected if they have different patterns
    # for (i, j), diff_var in pattern_diff_vars.items():
    #     model.Add(y[i] + y[j] <= diff_var + 1)

    return model, y, subsets, weights, cat_for_weight


def solve_model(model, y, weights):
    """
    Solve the model and return results.

    Args:
        model: CP-SAT model with constraints
        y: List of boolean variables
        weights: List of weights corresponding to variables

    Returns:
        tuple: (solver_status, chosen_indices, objective_value, solver)
    """
    # Set objective
    model.Maximize(sum(int(weights[i]) * y[i] for i in range(len(y))))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8  # Adjust to your CPU
    solver.parameters.max_time_in_seconds = 360.0  # Optional time cap

    status = solver.Solve(model)
    chosen = [i for i in range(len(y)) if solver.Value(y[i]) == 1 and weights[i] != 100000000]

    return status, chosen, solver.ObjectiveValue(), solver


def print_solution_board(design_goals_model: DesignGoalsModel, solver: cp_model.CpSolver):
    """
    Print the complete quilt board showing both color and pattern for each patch tile.

    Args:
        design_goals_model: The design goals model containing pattern and color variables
        solver: The solved CP-SAT solver containing solution values
    """
    from core.enums.color import Color
    from core.models.patch_tile import PatchTile

    print("\n" + "=" * 60)
    print("COMPLETE QUILT BOARD SOLUTION")
    print("=" * 60)

    # Create a quilt board to display the solution
    dummy_design_goals = [
        DesignGoalTiles.SIX_UNIQUE.value,
        DesignGoalTiles.THREE_PAIRS.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
    ]
    board = QuiltBoard(edge_setting=BOARD_CONFIG, design_goal_tiles=dummy_design_goals)

    # Get all pattern values for positions 0-48
    all_colors = list(Color)
    all_patterns = list(ALL_PATTERNS)

    # Fill the board with solved pattern and color values
    for pos in range(49):
        hex_pos = HexPosition.from_abs(pos)
        if not hex_pos.is_valid:
            continue

        # Skip edge positions and design goal positions as they're already set
        if hex_pos.is_edge_position() or hex_pos.is_design_goal_position():
            continue

        # Get pattern and color from solver if variables exist
        pattern_var_name = f"P_{pos}"
        color_var_name = f"C_{pos}"

        if pattern_var_name in design_goals_model.pattern_variables:
            pattern_value = solver.Value(design_goals_model.pattern_variables[pattern_var_name])
            pattern = all_patterns[pattern_value]

            if color_var_name in design_goals_model.color_variables:
                color_value = solver.Value(design_goals_model.color_variables[color_var_name])
                color = all_colors[color_value]

                # Create and place the patch tile
                patch_tile = PatchTile(color=color, pattern=pattern)
                board.tiles_by_pos[hex_pos] = patch_tile

    # Print the board using the built-in pretty print
    print(board.pretty_print())
    print()


def print_cat_patches_with_patterns(
    chosen_indices, subsets, weights, cat_for_weight, design_goals_model: DesignGoalsModel, solver: cp_model.CpSolver
):
    """
    Print each chosen cat patch subset with the pattern assigned to it.

    Args:
        chosen_indices: List of indices for chosen cat patches
        subsets: List of all subsets (patches)
        weights: List of weights for each subset
        cat_for_weight: Mapping from weight to cat name
        design_goals_model: The design goals model containing pattern variables
        solver: The solved CP-SAT solver containing solution values
    """
    print("\n" + "=" * 60)
    print("CHOSEN CAT PATCHES WITH PATTERNS")
    print("=" * 60)

    all_patterns = list(ALL_PATTERNS)

    # Group chosen patches by cat
    cat_patches = {}
    for idx in chosen_indices:
        weight = weights[idx]
        cat_name = cat_for_weight.get(weight, f"Unknown (weight {weight})")

        if cat_name not in cat_patches:
            cat_patches[cat_name] = []

        # Get the pattern for this patch (use first tile in subset as reference)
        subset = subsets[idx]
        if subset:  # Make sure subset is not empty
            # Find first tile that has a pattern variable
            pattern_info = "No pattern variable found"
            for tile_pos in subset:
                if tile_pos < 49:  # Valid board position
                    pattern_var_name = f"P_{tile_pos}"
                    if pattern_var_name in design_goals_model.pattern_variables:
                        pattern_value = solver.Value(design_goals_model.pattern_variables[pattern_var_name])
                        pattern = all_patterns[pattern_value]
                        pattern_info = f"{pattern.value}"
                        break

            cat_patches[cat_name].append({"subset": subset, "pattern": pattern_info, "weight": weight})

    # Print patches organized by cat
    for cat_name in sorted(cat_patches.keys()):
        print(f"\n{cat_name.upper()} PATCHES:")
        print("-" * 40)

        patches = cat_patches[cat_name]
        total_weight = sum(patch["weight"] for patch in patches)

        for i, patch in enumerate(patches, 1):
            subset_str = ", ".join(map(str, sorted(patch["subset"])))
            print(f"  Patch {i}: Tiles [{subset_str}] - Pattern: {patch['pattern']} - Weight: {patch['weight']}")

        print(f"  Total patches: {len(patches)} | Total weight: {total_weight}")

    # Print summary
    print("\nSUMMARY:")
    print(f"Total chosen patches: {len(chosen_indices)}")
    print(f"Cats represented: {len(cat_patches)}")
    total_objective = sum(weights[idx] for idx in chosen_indices)
    print(f"Total objective value: {total_objective}")
    print()


def main():
    """Example of how to use the model building process."""
    # Create a new model
    model = cp_model.CpModel()

    board_setting = EdgeTileSettings.BOARD_1
    m1 = DesignGoalTiles.TWO_TRIPLETS
    m3 = DesignGoalTiles.THREE_PAIRS
    m2 = DesignGoalTiles.FOUR_TWO

    base_model = design_goals_build_model(
        model, list(PATTERN_MAP.values()), board_setting, m1.value, m2.value, m3.value, cap=3, time_limit_s=200
    )

    # Choose cats (must be valid combination from different difficulty groups)
    cat_names = ["leo", "cira", "rumi"]  # Example: one from each difficulty group

    # Build the model with variables and constraints
    model, variables, subsets, weights, cat_for_weight = build_model(base_model, cat_names)

    # Solve the model
    status, chosen_indices, objective_value, solver = solve_model(model, variables, weights)

    # Print basic results
    print(f"Cats: {', '.join(cat_names)}")
    print(f"Solver status: {status}")
    print(f"Objective value: {objective_value}")
    print(f"Chosen patches: {len(chosen_indices)}")

    # Print detailed solution if we found a solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if chosen_indices:
            # Print the complete quilt board with colors and patterns
            print_solution_board(base_model, solver)

            # Print cat patches with their patterns
            print_cat_patches_with_patterns(chosen_indices, subsets, weights, cat_for_weight, base_model, solver)
        else:
            print("No patches were chosen in the solution.")
    else:
        print("No feasible solution found - cannot display board or patch details.")

    return status, chosen_indices, objective_value


if __name__ == "__main__":
    main()
