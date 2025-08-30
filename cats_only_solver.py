import json
import multiprocessing
from itertools import combinations
from pathlib import Path

from ortools.sat.python import cp_model

from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import ALL_PATTERNS, Pattern
from core.models.cat_scoring_tile import CatDifficultyGroup
from core.models.quilt_board import HexPosition, QuiltBoard

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

# Board configurations
BOARD_CONFIGS = {
    1: EdgeTileSettings.BOARD_1,
    2: EdgeTileSettings.BOARD_2,
    3: EdgeTileSettings.BOARD_3,
    4: EdgeTileSettings.BOARD_4,
}

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
    all_subsets = []
    all_weights = []
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


def build_graph_and_solve(cat_names, board_idx, log_file=None):
    """Build graph and solve for given cats and board configuration."""
    # Load patches for the specified cats
    subsets, weights, cat_for_weight = load_cat_patches(cat_names)

    if log_file:
        print(f"Testing cats: {', '.join(cat_names)} on board {board_idx}", file=log_file)
        print(f"Total patches: {len(subsets)}", file=log_file)
        for cat_name in cat_names:
            cat_patches = sum(1 for i, w in enumerate(weights) if w == CAT_WEIGHTS.get(cat_name, 1))
            print(f"{cat_name} patches: {cat_patches} (weight {CAT_WEIGHTS.get(cat_name, 1)})", file=log_file)

    # Build graph with vertices 0-48
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
        except Exception as e:
            if log_file:
                print(f"Error processing vertex {vertex}: {e}", file=log_file)

    if log_file:
        print(f"Built graph with {len(edges)} edges", file=log_file)

    # Add virtual vertices and edges
    add_virtual_vertices(edges, subsets, weights, board_idx)

    # Create edge position and pattern information for result printing
    dummy_design_goals = [
        DesignGoalTiles.SIX_UNIQUE.value,
        DesignGoalTiles.THREE_PAIRS.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
    ]
    board = QuiltBoard(edge_setting=BOARD_CONFIGS[board_idx], design_goal_tiles=dummy_design_goals)
    hex_positions = [HexPosition.from_abs(i) for i in range(49)]
    is_edge_positions = [pos.is_edge_position() for pos in hex_positions]
    patterns = [None if not pos.is_edge_position() else board.tiles_by_pos[pos].pattern for pos in hex_positions]

    # Call solve method
    result = solve(edges, subsets, weights, board_idx, cat_for_weight, log_file)

    # Add edge pattern information to the result
    return result + (is_edge_positions, patterns)


def add_virtual_vertices(vertex_edges, subsets, weights, board_idx):
    """Add virtual vertices for pattern constraints."""
    dummy_design_goals = [
        DesignGoalTiles.SIX_UNIQUE.value,
        DesignGoalTiles.THREE_PAIRS.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
    ]
    board = QuiltBoard(edge_setting=BOARD_CONFIGS[board_idx], design_goal_tiles=dummy_design_goals)

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


def solve(vertex_edges, subsets, weights, board_idx, cat_for_weight, log_file=None):
    """Solve the optimization problem."""
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
    board = QuiltBoard(edge_setting=BOARD_CONFIGS[board_idx], design_goal_tiles=dummy_design_goals)
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

    # Enumerate forbidden triangles within each same-label partition
    forbidden_triangles = []
    for w, idxs in same_label_idxs.items():
        # Skip building forbidden triangles for gwenivere
        if cat_for_weight.get(w) == "gwenivere":
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

    if log_file:
        print(f"Found {len(forbidden_triangles)} forbidden triangles", file=log_file)

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

    if log_file:
        print(f"Found {len(outer_pattern_restrictions)} outer pattern restrictions", file=log_file)

    # Build CP-SAT
    mdl = cp_model.CpModel()
    y = [mdl.NewBoolVar(f"y[{i}]") for i in range(m)]
    # Objective
    mdl.Maximize(sum(int(weights[i]) * y[i] for i in range(m)))

    # Disjointness per vertex
    for v in range(vars_count):
        mdl.Add(sum(y[i] for i in range(m) if (masks[i] >> v) & 1) <= 1)

    # Triangle cuts
    for i, j, k in forbidden_triangles:
        mdl.Add(y[i] + y[j] + y[k] <= 2)

    # Make sure that each cat occupy only 2 patterns
    for w, idxs in same_label_idxs.items():
        groups = pattern_groups[w]
        active = []
        for pattern, idxs in groups.items():
            b = mdl.NewBoolVar(f"b[{w}][{pattern.value}]")
            mdl.AddMaxEquality(b, [y[idx] for idx in idxs])
            active.append(b)
        mdl.Add(sum(active) <= 2)

    # Make sure that no two cats occupy the same pattern
    for i, j in outer_pattern_restrictions:
        mdl.Add(y[i] + y[j] <= 1)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8  # Adjust to your CPU
    solver.parameters.max_time_in_seconds = 360.0  # Optional time cap
    res = solver.Solve(mdl)

    chosen = [i for i in range(m) if solver.Value(y[i]) == 1 and weights[i] != 100000000]
    return res, chosen, solver.ObjectiveValue(), subsets, weights


def log_solution_details(
    log_file,
    result,
    chosen_subsets,
    objective_value,
    subsets,
    weights,
    is_edge_positions,
    patterns,
    cat_names,
    board_idx,
):
    """Log detailed solution information."""
    actual_score = int(objective_value) % 100
    print("\n=== SOLUTION SUMMARY ===", file=log_file)
    print(f"Cats: {', '.join(cat_names)}", file=log_file)
    print(f"Board: {board_idx}", file=log_file)
    print(f"Solution status: {result}", file=log_file)
    print(f"Raw objective value: {objective_value}", file=log_file)
    print(f"Actual score: {actual_score}", file=log_file)
    print(f"Number of chosen patches: {len(chosen_subsets)}", file=log_file)

    if chosen_subsets:
        print("\nChosen patch details:", file=log_file)
        for idx in chosen_subsets[:20]:  # Show first 20
            subset = subsets[idx]
            weight = weights[idx]

            # Check if this subset contains any edge tiles and get their patterns
            edge_patterns = []
            for vertex in subset:
                if vertex < len(is_edge_positions) and is_edge_positions[vertex]:
                    pattern = patterns[vertex]
                    if pattern and pattern not in edge_patterns:
                        edge_patterns.append(pattern)

            # Print subset info with pattern information if it has edge tiles
            if edge_patterns:
                pattern_names = [p.name for p in edge_patterns]
                print(
                    f"  Index {idx}: {subset} (weight {weight}) - Edge patterns: {', '.join(pattern_names)}",
                    file=log_file,
                )
            else:
                print(f"  Index {idx}: {subset} (weight {weight})", file=log_file)

        if len(chosen_subsets) > 20:
            print(f"  ... and {len(chosen_subsets) - 20} more patches", file=log_file)


def process_combination(args):
    """Worker function to process a single cat combination and board."""
    cat_combo, board_idx, log_dir, current_idx, total_combinations = args
    cat_names = sorted(cat_combo)  # Sort for consistent naming

    # Check if any log file for this combination already exists
    # We check with a pattern since we don't know the score yet
    pattern_files = list(log_dir.glob(f"{board_idx}_*_{'_'.join(cat_names)}.log"))

    if pattern_files:
        existing_file = pattern_files[0].name
        return (
            f"[{current_idx}/{total_combinations}] Skipping: {', '.join(cat_names)} on board {board_idx} "
            f"(already exists: {existing_file})"
        )

    print(f"Testing {', '.join(cat_names)} on board {board_idx}")
    try:
        # Run the solver
        result, chosen_subsets, objective_value, subsets, weights, is_edge_positions, patterns = build_graph_and_solve(
            cat_names, board_idx
        )

        # Calculate actual score (remove artificial component)
        actual_score = int(objective_value) % 100

        # Create log filename
        log_filename = f"{board_idx}_{actual_score:02d}_{'_'.join(cat_names)}.log"
        log_path = log_dir / log_filename

        # Write detailed log
        with open(log_path, "w") as log_file:
            log_solution_details(
                log_file,
                result,
                chosen_subsets,
                objective_value,
                subsets,
                weights,
                is_edge_positions,
                patterns,
                cat_names,
                board_idx,
            )

        return (
            f"[{current_idx}/{total_combinations}] Completed: {', '.join(cat_names)} on board {board_idx} -> "
            f"Score: {actual_score}, Status: {result}, Log: {log_filename}"
        )

    except Exception as e:
        error_msg = f"Error testing {', '.join(cat_names)} on board {board_idx}: {str(e)}"

        # Still create a log file to record the error
        log_filename = f"{board_idx}_00_{'_'.join(cat_names)}.log"
        log_path = log_dir / log_filename

        with open(log_path, "w") as log_file:
            print(f"ERROR: {error_msg}", file=log_file)
            print(f"Cats: {', '.join(cat_names)}", file=log_file)
            print(f"Board: {board_idx}", file=log_file)

        return f"[{current_idx}/{total_combinations}] ERROR: {', '.join(cat_names)} on board {board_idx} -> {error_msg}"


def main():
    """Main function to test all combinations of 3 cats across all boards using multiprocessing."""
    log_dir = Path("logs/cat_solutions")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    # Leave one core free for system processes, but use at least 1 core
    num_processes = max(1, num_cores - 6)

    print("Starting comprehensive cat combination testing...")
    print(f"Available cats: {', '.join(AVAILABLE_CATS)}")
    print("Testing all valid combinations of 3 cats across boards 1-4")
    print("Note: Cats from the same difficulty group cannot be combined together")
    print(f"Using {num_processes} parallel processes (detected {num_cores} CPU cores)")

    # Show difficulty groups
    print("\nDifficulty groups:")
    for group in [CatDifficultyGroup.ONE_DOT, CatDifficultyGroup.TWO_DOT, CatDifficultyGroup.THREE_DOT]:
        cats_in_group = [cat for cat, cat_group in CAT_DIFFICULTY_GROUPS.items() if cat_group == group]
        print(f"  {group.name} ({group.value} dot): {', '.join(sorted(cats_in_group))}")

    # Filter out invalid combinations (all from same difficulty group)
    all_combinations = list(combinations(AVAILABLE_CATS, 3))
    valid_combinations = [combo for combo in all_combinations if is_valid_cat_combination(combo)]

    print(f"Total possible combinations: {len(all_combinations)}")
    print(f"Valid combinations (different difficulty groups): {len(valid_combinations)}")

    # Create all tasks (cat_combo, board_idx) pairs
    tasks = []
    current_idx = 0
    for cat_combo in valid_combinations:
        for board_idx in range(1, 2):
            current_idx += 1
            tasks.append((cat_combo, board_idx, log_dir, current_idx, len(valid_combinations) * 4))

    total_combinations = len(tasks)
    print(f"Total combinations to test: {total_combinations}")

    # Process all tasks using multiprocessing
    print("\nStarting parallel processing...")

    try:
        with multiprocessing.Pool(num_processes) as pool:
            # Use imap for real-time progress updates
            results = []
            for result in pool.imap(process_combination, tasks):
                results.append(result)
                print(result)

        print(f"\nCompleted all {total_combinations} combinations!")
        print(f"Log files saved to: {log_dir}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        pool.terminate()
        pool.join()
    except Exception as e:
        print(f"\nError during multiprocessing: {e}")
        print("Falling back to sequential processing...")

        # Fallback to sequential processing if multiprocessing fails
        for task in tasks:
            result = process_combination(task)
            print(result)


if __name__ == "__main__":
    main()
