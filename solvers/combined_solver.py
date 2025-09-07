"""
Combined solver that integrates design goals, cats, and buttons constraints.

This solver creates a unified model that optimizes for:
1. Design goal tiles satisfaction (base constraints)
2. Cat patches placement and scoring
3. Button patterns and rainbow scoring

Parameters:
- Choice of 3 design goal tiles
- Choice of 3 cat tiles
- Board configuration
"""

from datetime import datetime

from ortools.sat.python import cp_model

from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import PATTERN_MAP
from core.models.design_goal_tile import DesignGoalTile
from core.models.quilt_board import QuiltBoard
from solvers.buttons_solver import ButtonsModelComponents, add_buttons_constraints
from solvers.cats_modeler import CatsModelComponents, add_cats_constraints, is_valid_cat_combination
from solvers.restructured_design_goals_solver import DesignGoalsModel
from solvers.restructured_design_goals_solver import build_model as design_goals_build_model


class CombinedModelComponents:
    """Container for all components of the combined solver."""

    def __init__(
        self,
        design_goals_model: DesignGoalsModel,
        cats_components: CatsModelComponents,
        buttons_components: ButtonsModelComponents,
        board: QuiltBoard,
        cat_names: list[str],
    ):
        self.design_goals_model = design_goals_model
        self.cats_components = cats_components
        self.buttons_components = buttons_components
        self.board = board
        self.cat_names = cat_names


def build_combined_model(
    design_goal_tiles: tuple[DesignGoalTile, DesignGoalTile, DesignGoalTile],
    cat_names: tuple[str, str, str],
    board_setting: EdgeTileSettings = EdgeTileSettings.BOARD_1,
    cap: int = 3,
    time_limit_s: float = 300.0,
    cats_weight: float = 1.0,
    buttons_weight: float = 3.0,
    missing_pattern: int = None,
    missing_color: int = None,
) -> CombinedModelComponents:
    """
    Build a combined model with design goals, cats, and buttons constraints.

    Args:
        design_goal_tiles: Tuple of 3 design goal tiles
        cat_names: Tuple of 3 cat names
        board_setting: Board edge configuration
        cap: Maximum allowed count for any ordered pair across positions
        time_limit_s: Time limit for solving in seconds
        cats_weight: Weight for cats objective (default 1.0)
        buttons_weight: Weight for buttons objective (default 1.0)
        missing_pattern: Which design goal (1, 2, or 3) should not satisfy pattern constraints
        missing_color: Which design goal (1, 2, or 3) should not satisfy color constraints

    Returns:
        CombinedModelComponents: Container with all model components
    """
    # Validate inputs
    if not is_valid_cat_combination(list(cat_names)):
        raise ValueError(f"Invalid cat combination: {cat_names}. Cats must be from different difficulty groups.")

    # Create base design goals model
    model = cp_model.CpModel()
    v = list(PATTERN_MAP.values())

    m1, m2, m3 = design_goal_tiles
    design_goals_model = design_goals_build_model(
        model,
        v,
        board_setting,
        m1,
        m2,
        m3,
        cap=cap,
        time_limit_s=time_limit_s,
        missing_pattern=missing_pattern,
        missing_color=missing_color,
    )

    # Add cats constraints
    print("Adding cats constraints...")
    cats_components = add_cats_constraints(design_goals_model, list(cat_names))

    # Add buttons constraints
    print("Adding buttons constraints...")
    buttons_components = add_buttons_constraints(design_goals_model, board_setting)

    # Create combined objective function
    cats_objective = sum(
        int(cats_components.weights[i]) * cats_components.y_variables[i]
        for i in range(len(cats_components.y_variables))
    )

    buttons_objective = (
        sum(
            buttons_components.r[_r][_k]
            for _r in range(len(buttons_components.r))
            for _k in range(len(buttons_components.r[0]))
        )
        + buttons_components.bonus
    )

    # Combined weighted objective
    total_objective = cats_weight * cats_objective + buttons_weight * buttons_objective

    # Set the combined objective
    design_goals_model.model.Maximize(total_objective)

    # Create board for reference
    board = QuiltBoard(edge_setting=board_setting, design_goal_tiles=[m1, m2, m3])

    return CombinedModelComponents(
        design_goals_model=design_goals_model,
        cats_components=cats_components,
        buttons_components=buttons_components,
        board=board,
        cat_names=list(cat_names),
    )


def get_cat_patch_pattern(subset: list, solved_patterns: dict) -> str:
    """
    Get pattern information for a cat patch subset.

    Args:
        subset: List of absolute positions in the cat patch
        solved_patterns: Dictionary mapping pattern variables to their values

    Returns:
        String describing the patterns in this patch
    """
    from core.enums.pattern import ALL_PATTERNS

    if not solved_patterns:
        return "Unknown"

    patterns = []
    for pos in sorted(subset):
        pattern_var_name = f"P_{pos}"
        if pattern_var_name in solved_patterns:
            pattern_value = solved_patterns[pattern_var_name]
            pattern = ALL_PATTERNS[pattern_value]
            return pattern.name
        patterns.append("Unknown")

    # Group by pattern and show counts
    from collections import Counter

    pattern_counts = Counter(patterns)
    pattern_summary = ", ".join(f"{count}x {pattern}" for pattern, count in pattern_counts.items())

    return pattern_summary


def calculate_design_goals_score(
    design_goal_tiles: list,
    missing_pattern: int = None,
    missing_color: int = None,
) -> int:
    """
    Calculate the total score for design goals based on missing pattern/color.

    Args:
        design_goal_tiles: List of 3 design goal tiles
        missing_pattern: Which design goal (1, 2, or 3) is missing pattern satisfaction
        missing_color: Which design goal (1, 2, or 3) is missing color satisfaction

    Returns:
        Total design goals score
    """
    total_score = 0

    for i, tile in enumerate(design_goal_tiles):
        design_goal_index = i + 1  # 1-based indexing

        # Check if this design goal has missing pattern or color
        has_missing_pattern = missing_pattern == design_goal_index
        has_missing_color = missing_color == design_goal_index

        # If either pattern or color is missing, use lower_points (partial score)
        # Otherwise use higher_points (full score for both pattern and color)
        if has_missing_pattern or has_missing_color:
            tile_score = tile.lower_points
        else:
            tile_score = tile.higher_points

        total_score += tile_score

    return total_score


def solve_combined_model(
    combined_components: CombinedModelComponents,
    time_limit_sec: float = 300.0,
    workers: int = 8,
    missing_pattern: int = None,
    missing_color: int = None,
) -> dict:
    """
    Solve the combined model and return comprehensive results.

    Args:
        combined_components: Combined model components
        time_limit_sec: Time limit for solving
        workers: Number of worker threads
        missing_pattern: Which design goal (1, 2, or 3) should not satisfy pattern constraints
        missing_color: Which design goal (1, 2, or 3) should not satisfy color constraints

    Returns:
        Dictionary containing all solution information
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = workers

    print(f"Solving combined model with {workers} workers, {time_limit_sec}s time limit...")
    status = solver.Solve(combined_components.design_goals_model.model)

    result = {"status": solver.StatusName(status)}

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Calculate the modified score: base score (% 100000) + design goals scores
        raw_objective_value = solver.ObjectiveValue()
        base_score = int(raw_objective_value) % 100000

        # Calculate design goals scores
        design_goals_score = calculate_design_goals_score(
            combined_components.design_goals_model.design_goal_tiles, missing_pattern, missing_color
        )

        # Final modified score
        modified_objective_value = base_score + design_goals_score

        result["raw_objective_value"] = raw_objective_value
        result["objective_value"] = modified_objective_value
        result["base_score"] = base_score
        result["design_goals_score"] = design_goals_score
        result["solve_time"] = solver.WallTime()

        # Extract cats solution
        cats_chosen = [
            i
            for i in range(len(combined_components.cats_components.y_variables))
            if solver.Value(combined_components.cats_components.y_variables[i]) == 1
            and combined_components.cats_components.weights[i] != 100000000
        ]
        cats_objective_value = sum(combined_components.cats_components.weights[i] for i in cats_chosen)

        result["cats_chosen_indices"] = cats_chosen
        result["cats_objective_value"] = cats_objective_value
        result["cats_count"] = len(cats_chosen)

        # Extract buttons solution
        buttons_representatives = []
        for _r in range(len(combined_components.buttons_components.r)):
            for _k in range(len(combined_components.buttons_components.r[0])):
                if solver.Value(combined_components.buttons_components.r[_r][_k]) == 1:
                    buttons_representatives.append((_r, _k))

        buttons_bonus = solver.Value(combined_components.buttons_components.bonus)
        buttons_objective_value = len(buttons_representatives) + buttons_bonus

        result["buttons_representatives"] = buttons_representatives
        result["buttons_bonus"] = buttons_bonus
        result["buttons_objective_value"] = buttons_objective_value

        # Extract color and pattern solutions
        solved_colors = {}
        solved_patterns = {}

        for var_name, var in combined_components.design_goals_model.color_variables.items():
            solved_colors[var_name] = solver.Value(var)

        for var_name, var in combined_components.design_goals_model.pattern_variables.items():
            solved_patterns[var_name] = solver.Value(var)

        result["solved_colors"] = solved_colors
        result["solved_patterns"] = solved_patterns

        # Add metadata
        result["cat_names"] = combined_components.cat_names
        result["design_goal_names"] = [
            dg.config_name for dg in combined_components.design_goals_model.design_goal_tiles
        ]
        result["board_setting"] = combined_components.board.edge_setting.name

    else:
        print(f"No solution found. Status: {solver.StatusName(status)}")

    return result


def print_combined_solution(
    combined_components: CombinedModelComponents, result: dict, missing_pattern: int = None, missing_color: int = None
):
    """
    Print comprehensive solution details for the combined model.

    Args:
        combined_components: Combined model components
        result: Solution result dictionary
        missing_pattern: Which design goal (1, 2, or 3) should not satisfy pattern constraints
        missing_color: Which design goal (1, 2, or 3) should not satisfy color constraints
    """
    print("\n" + "=" * 80)
    print("COMBINED SOLVER SOLUTION")
    print("=" * 80)

    # Print configuration
    print(f"Board: {result['board_setting']}")
    print(f"Design Goals: {' -> '.join(result['design_goal_names'])}")
    print(f"Cats: {', '.join(result['cat_names'])}")
    print(f"Status: {result['status']}")

    if result.get("objective_value") is not None:
        print(f"Total Objective Value: {result['objective_value']}")
        if "raw_objective_value" in result:
            print(f"  Raw Objective Value: {result['raw_objective_value']}")
            print(f"  Base Score (% 100000): {result['base_score']}")
            print(f"  Design Goals Score: {result['design_goals_score']}")
        print(f"Solve Time: {result.get('solve_time', 'N/A')}s")
        print()

        # Design Goals results
        print("DESIGN GOALS RESULTS:")
        print("-" * 40)
        print(f"Total Design Goals Score: {result.get('design_goals_score', 0)}")
        print()

        design_goal_tiles = combined_components.design_goals_model.design_goal_tiles
        for i, tile in enumerate(design_goal_tiles):
            goal_num = i + 1
            has_missing_pattern = missing_pattern == goal_num
            has_missing_color = missing_color == goal_num

            if has_missing_pattern or has_missing_color:
                score = tile.lower_points
                status = f"PARTIAL ({'missing pattern' if has_missing_pattern else 'missing color'})"
            else:
                score = tile.higher_points
                status = "COMPLETE (both pattern and color)"

            print(f"  Goal {goal_num} ({tile.config_name}): {score} points - {status}")

        print()

        # Cats results
        print("CATS RESULTS:")
        print("-" * 40)
        print(f"Cats Objective: {result['cats_objective_value']}")
        print(f"Patches Selected: {result['cats_count']}")

        if result["cats_chosen_indices"]:
            # Group by cat
            cat_patches = {}
            for idx in result["cats_chosen_indices"]:
                weight = combined_components.cats_components.weights[idx]
                cat_name = combined_components.cats_components.cat_for_weight.get(weight, f"Unknown (weight {weight})")

                if cat_name not in cat_patches:
                    cat_patches[cat_name] = []

                subset = combined_components.cats_components.subsets[idx]
                cat_patches[cat_name].append({"subset": subset, "weight": weight})

            for cat_name in sorted(cat_patches.keys()):
                patches = cat_patches[cat_name]
                total_weight = sum(patch["weight"] for patch in patches)
                print(f"  {cat_name}: {len(patches)} patches, weight {total_weight}")
                for i, patch in enumerate(patches, 1):
                    subset_str = ", ".join(map(str, sorted(patch["subset"])))
                    # Get pattern info for this patch
                    pattern_info = get_cat_patch_pattern(patch["subset"], result.get("solved_patterns", {}))
                    print(f"    Patch {i}: [{subset_str}] (weight {patch['weight']}) - Pattern: {pattern_info}")

        print()

        # Buttons results
        print("BUTTONS RESULTS:")
        print("-" * 40)
        print(f"Buttons Objective: {result['buttons_objective_value']}")
        print(f"K-consistent Components: {len(result['buttons_representatives'])}")
        print(f"Rainbow Bonus: {'Yes' if result['buttons_bonus'] else 'No'}")

        if result["buttons_representatives"]:
            from core.enums.color import Color

            color_names = list(Color.__members__.keys())

            print("Representative Components:")
            for idx, (subset_idx, color_idx) in enumerate(result["buttons_representatives"], 1):
                tile_positions = combined_components.buttons_components.tile_sets[subset_idx]
                color_name = color_names[color_idx]
                tile_positions_str = ", ".join(str(pos.abs) for pos in tile_positions)
                print(f"  Component {idx}: Subset {subset_idx} with {color_name}")
                print(f"    Tiles (absolute positions): {tile_positions_str}")

        print()

        # Board visualization
        print("QUILT BOARD:")
        print("-" * 40)
        # Create solution board similar to other solvers
        solution_board = create_solution_board(combined_components, result)
        print(solution_board.pretty_print())

    else:
        print("No solution found.")

    print("\n" + "=" * 80)


def create_solution_board(combined_components: CombinedModelComponents, result: dict) -> QuiltBoard:
    """
    Create a QuiltBoard showing the complete solution.

    Args:
        combined_components: Combined model components
        result: Solution result dictionary

    Returns:
        QuiltBoard with solved colors and patterns
    """
    from core.enums.color import ALL_COLORS
    from core.enums.pattern import ALL_PATTERNS
    from core.models.patch_tile import PatchTile

    # Create solution board
    solution_board = QuiltBoard(
        edge_setting=combined_components.board.edge_setting,
        design_goal_tiles=combined_components.design_goals_model.design_goal_tiles,
    )

    # Get solved colors and patterns
    solved_colors = result.get("solved_colors", {})
    solved_patterns = result.get("solved_patterns", {})

    # Fill in patch tiles
    patch_tiles = combined_components.board.get_all_patch_tiles()

    for patch_pos in patch_tiles:
        abs_pos = patch_pos.abs
        color_var_name = f"C_{abs_pos}"
        pattern_var_name = f"P_{abs_pos}"

        # Get solved color
        if color_var_name in solved_colors:
            color_value = solved_colors[color_var_name]
            solved_color = ALL_COLORS[color_value]  # Convert from 1-based to 0-based
        else:
            solved_color = ALL_COLORS[0]  # Fallback

        # Get solved pattern
        if pattern_var_name in solved_patterns:
            pattern_value = solved_patterns[pattern_var_name]
            solved_pattern = ALL_PATTERNS[pattern_value]  # Convert from 1-based to 0-based
        else:
            solved_pattern = ALL_PATTERNS[0]  # Fallback

        # Create patch tile
        patch_tile = PatchTile(color=solved_color, pattern=solved_pattern)
        solution_board.tiles_by_pos[patch_pos] = patch_tile

    return solution_board


def save_combined_result_to_log(
    combined_components: CombinedModelComponents,
    best_result: dict,
    all_results: list,
    missing_pattern: int = None,
    missing_color: int = None,
) -> str:
    """
    Save the combined result to a log file.

    Args:
        combined_components: Combined model components
        best_result: Best solution result dictionary
        all_results: List of all configuration results for comparison table
        missing_pattern: Which design goal (1, 2, or 3) should not satisfy pattern constraints
        missing_color: Which design goal (1, 2, or 3) should not satisfy color constraints

    Returns:
        Filename of the created log file
    """
    # Create filename with 3-digit padded best score
    best_score = best_result.get("objective_value", 0)
    score_str = f"{best_score:03d}"  # 3-digit with leading zeros
    board_name = best_result["board_setting"]
    design_goal_str = "_".join(best_result["design_goal_names"])
    cats_str = "_".join(best_result["cat_names"])
    filename = f"logs/overall_solutions/{score_str}_{board_name}_{design_goal_str}_{cats_str}.log"

    with open(filename, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("CALICO COMBINED SOLVER RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Board: {best_result['board_setting']}\n")
        f.write(f"Design Goals: {' -> '.join(best_result['design_goal_names'])}\n")
        f.write(f"Cats: {', '.join(best_result['cat_names'])}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write configuration comparison table
        f.write("CONFIGURATION COMPARISON TABLE\n")
        f.write("=" * 80 + "\n")

        # Check if any configurations were skipped based on all_results length
        if len(all_results) < 7:
            f.write("Note: Missing color configurations were skipped due to optimal solution with max button score.\n")
            f.write("Only configurations 1-4 (no missing + missing patterns) were tested.\n\n")

        f.write(f"{'#':<3} {'Configuration':<35} {'Status':<12} {'Score':<10} {'Best':<6}\n")
        f.write("-" * 70 + "\n")

        best_score = best_result.get("objective_value", -1)
        for i, result in enumerate(all_results, 1):
            config_name = result.get("config_name", "Unknown")[:34]
            status = result.get("status", "Unknown")[:11]

            if result.get("objective_value") is not None:
                score_str = str(result["objective_value"])
                is_best = "YES" if result["objective_value"] == best_score else ""
            else:
                score_str = "N/A"
                is_best = ""

            f.write(f"{i:<3} {config_name:<35} {status:<12} {score_str:<10} {is_best:<6}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Write solver results for best configuration
        f.write("BEST CONFIGURATION DETAILS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Status: {best_result['status']}\n")

        if best_result.get("objective_value") is not None:
            f.write(f"Total Objective Value: {best_result['objective_value']}\n")
            if "raw_objective_value" in best_result:
                f.write(f"Raw Objective Value: {best_result['raw_objective_value']}\n")
                f.write(f"Base Score (% 100000): {best_result['base_score']}\n")
                f.write(f"Design Goals Score: {best_result['design_goals_score']}\n")
            f.write(f"Solve Time: {best_result.get('solve_time', 'N/A')}s\n\n")

            # Write detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 60 + "\n")

            # Design Goals section
            f.write("DESIGN GOALS RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Design Goals Score: {best_result.get('design_goals_score', 0)}\n\n")

            design_goal_tiles = combined_components.design_goals_model.design_goal_tiles
            for i, tile in enumerate(design_goal_tiles):
                goal_num = i + 1
                has_missing_pattern = missing_pattern == goal_num
                has_missing_color = missing_color == goal_num

                if has_missing_pattern or has_missing_color:
                    score = tile.lower_points
                    status = f"PARTIAL ({'missing pattern' if has_missing_pattern else 'missing color'})"
                else:
                    score = tile.higher_points
                    status = "COMPLETE (both pattern and color)"

                f.write(f"  Goal {goal_num} ({tile.config_name}): {score} points - {status}\n")

            f.write("\n")

            # Cats section
            f.write("CATS RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cats Objective: {best_result['cats_objective_value']}\n")
            f.write(f"Patches Selected: {best_result['cats_count']}\n")

            if best_result["cats_chosen_indices"]:
                # Group by cat (similar to print function)
                cat_patches = {}
                for idx in best_result["cats_chosen_indices"]:
                    weight = combined_components.cats_components.weights[idx]
                    cat_name = combined_components.cats_components.cat_for_weight.get(
                        weight, f"Unknown (weight {weight})"
                    )

                    if cat_name not in cat_patches:
                        cat_patches[cat_name] = []

                    subset = combined_components.cats_components.subsets[idx]
                    cat_patches[cat_name].append({"subset": subset, "weight": weight})

                for cat_name in sorted(cat_patches.keys()):
                    patches = cat_patches[cat_name]
                    total_weight = sum(patch["weight"] for patch in patches)
                    f.write(f"  {cat_name}: {len(patches)} patches, weight {total_weight}\n")
                    for i, patch in enumerate(patches, 1):
                        subset_str = ", ".join(map(str, sorted(patch["subset"])))
                        # Get pattern info for this patch
                        pattern_info = get_cat_patch_pattern(patch["subset"], best_result.get("solved_patterns", {}))
                        f.write(f"    Patch {i}: [{subset_str}] (weight {patch['weight']}) - Pattern: {pattern_info}\n")

            f.write("\n")

            # Buttons section
            f.write("BUTTONS RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Buttons Objective: {best_result['buttons_objective_value']}\n")
            f.write(f"K-consistent Components: {len(best_result['buttons_representatives'])}\n")
            f.write(f"Rainbow Bonus: {'Yes' if best_result['buttons_bonus'] else 'No'}\n")

            if best_result["buttons_representatives"]:
                from core.enums.color import Color

                color_names = list(Color.__members__.keys())

                f.write("Representative Components:\n")
                for idx, (subset_idx, color_idx) in enumerate(best_result["buttons_representatives"], 1):
                    tile_positions = combined_components.buttons_components.tile_sets[subset_idx]
                    color_name = color_names[color_idx]
                    tile_positions_str = ", ".join(str(pos.abs) for pos in tile_positions)
                    f.write(f"  Component {idx}: Subset {subset_idx} with {color_name}\n")
                    f.write(f"    Tiles (absolute positions): {tile_positions_str}\n")

            f.write("\n")

            # Board solution
            f.write("QUILT BOARD SOLUTION:\n")
            f.write("-" * 60 + "\n")
            solution_board = create_solution_board(combined_components, best_result)
            f.write(solution_board.pretty_print() + "\n\n")

        else:
            f.write("No solution found.\n\n")

        f.write("=" * 80 + "\n")

    return filename


def run_optimization_for_config(
    design_goal_tiles: tuple, cat_names: tuple, board_setting: EdgeTileSettings = EdgeTileSettings.BOARD_1
) -> tuple:
    """
    Run the combined solver with 7 different missing_pattern/missing_color configurations.

    Args:
        design_goal_tiles: Tuple of 3 DesignGoalTile objects
        cat_names: Tuple of 3 cat name strings
        board_setting: Board edge configuration

    Returns:
        Tuple of (all_results, best_result, best_config)
    """
    print("Running combined solver with 7 different configurations...")
    print("=" * 80)

    # Define 7 configurations to test
    configurations = [
        {"missing_pattern": None, "missing_color": None, "name": "No missing (complete)"},
        {"missing_pattern": 1, "missing_color": None, "name": "Missing pattern for goal 1"},
        {"missing_pattern": 2, "missing_color": None, "name": "Missing pattern for goal 2"},
        {"missing_pattern": 3, "missing_color": None, "name": "Missing pattern for goal 3"},
        {"missing_pattern": None, "missing_color": 1, "name": "Missing color for goal 1"},
        {"missing_pattern": None, "missing_color": 2, "name": "Missing color for goal 2"},
        {"missing_pattern": None, "missing_color": 3, "name": "Missing color for goal 3"},
    ]

    # Track if we should skip missing color configurations
    skip_missing_color = False

    all_results = []
    best_result = None
    best_config = None
    best_components = None
    best_score = -1

    # Run each configuration
    for i, config in enumerate(configurations, 1):
        # Skip missing color configurations if we already found optimal solution with max button score
        if skip_missing_color and config["missing_color"] is not None:
            print(f"\nSkipping Configuration {i}/7: {config['name']} (already found optimal with max button score)")
            continue

        total_configs = 7 if not skip_missing_color else 4
        print(f"\nConfiguration {i}/{total_configs}: {config['name']}")
        print(f"missing_pattern: {config['missing_pattern']}, missing_color: {config['missing_color']}")
        print("-" * 60)

        try:
            # Build model for this configuration
            combined_components = build_combined_model(
                design_goal_tiles=design_goal_tiles,
                cat_names=cat_names,
                board_setting=board_setting,
                cats_weight=1.0,
                buttons_weight=3.0,
                time_limit_s=300.0,
                missing_pattern=config["missing_pattern"],
                missing_color=config["missing_color"],
            )

            # Solve model
            result = solve_combined_model(
                combined_components,
                time_limit_sec=300.0,
                workers=8,
                missing_pattern=config["missing_pattern"],
                missing_color=config["missing_color"],
            )

            # Add configuration info to result
            result["config_name"] = config["name"]
            result["missing_pattern"] = config["missing_pattern"]
            result["missing_color"] = config["missing_color"]

            all_results.append(result)

            # Check if this is the best result so far
            if result.get("objective_value") is not None:
                current_score = result["objective_value"]
                print(f"Final Score: {current_score}")
                print(f"Status: {result['status']}")
                print(f"Execution time: {result.get('solve_time', 'N/A')}s")

                if current_score > best_score:
                    best_score = current_score
                    best_result = result
                    best_config = config
                    best_components = combined_components
                    print("*** NEW BEST SCORE! ***")

                # Print detailed solution for this configuration
                print_combined_solution(combined_components, result, config["missing_pattern"], config["missing_color"])

                # Check if this is the first configuration (no missing) with optimal solution and max button score
                if (
                    i == 1 and result.get("status") == "OPTIMAL" and result.get("buttons_objective_value", 0) >= 11
                ):  # Max button score is 11 patterns * 3 = 33
                    skip_missing_color = True
                    print("*** OPTIMAL SOLUTION WITH MAXIMUM BUTTON SCORE FOUND! ***")
                    print("*** Skipping missing color configurations as they cannot improve the score ***")
            else:
                print("No solution found for this configuration")

        except Exception as e:
            print(f"Error in configuration {i}: {e}")
            continue

    # Print summary of all results
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("=" * 80)

    if skip_missing_color:
        print(
            "Note: Missing color configurations (5-7) were skipped due to optimal solution with max button score found."
        )
        print("Only configurations 1-4 (no missing + missing patterns) were tested.")
        print()

    print(f"{'#':<3} {'Configuration':<35} {'Status':<12} {'Score':<10} {'Best':<6}")
    print("-" * 70)

    for i, result in enumerate(all_results, 1):
        config_name = result.get("config_name", "Unknown")[:34]
        status = result.get("status", "Unknown")[:11]

        if result.get("objective_value") is not None:
            score_str = str(result["objective_value"])
            is_best = "YES" if result["objective_value"] == best_score else ""
        else:
            score_str = "N/A"
            is_best = ""

        print(f"{i:<3} {config_name:<35} {status:<12} {score_str:<10} {is_best:<6}")

    # Print best configuration details
    if best_result is not None:
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION DETAILS")
        print("=" * 80)
        print(f"Configuration: {best_config['name']}")
        print(f"missing_pattern: {best_config['missing_pattern']}")
        print(f"missing_color: {best_config['missing_color']}")
        print(f"Best Score: {best_score}")

        print("\nBreakdown:")
        print(f"  Raw Objective Value: {best_result.get('raw_objective_value', 'N/A')}")
        print(f"  Base Score (% 100000): {best_result.get('base_score', 'N/A')}")
        print(f"  Design Goals Score: {best_result.get('design_goals_score', 'N/A')}")
        print(f"  Cats Objective: {best_result.get('cats_objective_value', 'N/A')}")
        print(f"  Buttons Objective: {best_result.get('buttons_objective_value', 'N/A')}")

        # Save only the best result to log
        if best_components is not None:
            log_filename = save_combined_result_to_log(
                best_components, best_result, all_results, best_config["missing_pattern"], best_config["missing_color"]
            )
            print(f"\nBest result saved to: {log_filename}")

    else:
        print("\nNo valid solutions found in any configuration!")

    print("=" * 80)
    return all_results, best_result, best_config


def main():
    """Run the combined solver with 7 different missing_pattern/missing_color configurations."""
    # Default configuration
    design_goals = (DesignGoalTiles.FOUR_TWO.value, DesignGoalTiles.SIX_UNIQUE.value, DesignGoalTiles.THREE_PAIRS.value)
    cat_names = ("leo", "cira", "rumi")  # One from each difficulty group
    board_setting = EdgeTileSettings.BOARD_1

    return run_optimization_for_config(design_goals, cat_names, board_setting)


if __name__ == "__main__":
    main()
