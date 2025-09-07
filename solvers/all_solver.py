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
from solvers.objective_design_goals_solver import ObjectiveDesignGoalsModel, build_objective_model


class CombinedModelComponents:
    """Container for all components of the combined solver."""

    def __init__(
        self,
        design_goals_model: ObjectiveDesignGoalsModel,
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
    design_goals_weight: float = 1.0,
    cats_weight: float = 1.0,
    buttons_weight: float = 3.0,
) -> CombinedModelComponents:
    """
    Build a combined model with design goals, cats, and buttons constraints.

    Args:
        design_goal_tiles: Tuple of 3 design goal tiles
        cat_names: Tuple of 3 cat names
        board_setting: Board edge configuration
        cap: Maximum allowed count for any ordered pair across positions
        time_limit_s: Time limit for solving in seconds
        design_goals_weight: Weight for design goals objective (default 1.0)
        cats_weight: Weight for cats objective (default 1.0)
        buttons_weight: Weight for buttons objective (default 3.0)

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
    design_goals_model = build_objective_model(model, v, m1, m2, m3, cap, time_limit_s, board_setting)

    # Add cats constraints
    print("Adding cats constraints...")
    cats_components = add_cats_constraints(design_goals_model, list(cat_names))

    # Add buttons constraints
    print("Adding buttons constraints...")
    buttons_components = add_buttons_constraints(design_goals_model, board_setting)

    # Create combined objective function
    design_goals_objective = sum(design_goals_model.design_goal_scores.values())

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
    total_objective = (
        design_goals_weight * design_goals_objective + cats_weight * cats_objective + buttons_weight * buttons_objective
    )

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


def solve_combined_model(
    combined_components: CombinedModelComponents,
    time_limit_sec: float = 300.0,
    workers: int = 8,
) -> dict:
    """
    Solve the combined model and return comprehensive results.

    Args:
        combined_components: Combined model components
        time_limit_sec: Time limit for solving
        workers: Number of worker threads

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
        result["objective_value"] = solver.ObjectiveValue()
        result["objective_value_mod"] = solver.ObjectiveValue() % 100000
        result["solve_time"] = solver.WallTime()

        # Extract design goals solution
        design_goal_scores = {}
        total_design_goals_score = 0
        for name, var in combined_components.design_goals_model.design_goal_scores.items():
            score = solver.Value(var)
            design_goal_scores[name] = score
            total_design_goals_score += score
        result["design_goal_scores"] = design_goal_scores
        result["total_design_goals_score"] = total_design_goals_score

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


def print_combined_solution(combined_components: CombinedModelComponents, result: dict):
    """
    Print comprehensive solution details for the combined model.

    Args:
        combined_components: Combined model components
        result: Solution result dictionary
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
        print(f"Total Objective Value: {result['objective_value']} (mod 100000: {result['objective_value_mod']})")
        print(f"Solve Time: {result.get('solve_time', 'N/A')}s")
        print()

        # Design Goals results
        print("DESIGN GOALS RESULTS:")
        print("-" * 40)
        print(f"Design Goals Score: {result['total_design_goals_score']}")
        for i, (_goal_name, score) in enumerate(result["design_goal_scores"].items()):
            goal_tile = combined_components.design_goals_model.design_goal_tiles[i]
            if score == goal_tile.higher_points:
                status = "FULL (Pattern + Color)"
            elif score == goal_tile.lower_points:
                status = "PARTIAL (Pattern or Color)"
            else:
                status = "NONE"
            print(f"  Goal {i + 1} ({goal_tile.config_name}): {score}/{goal_tile.higher_points} points - {status}")
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

                # Get patterns for this patch
                patch_patterns = []
                for abs_pos in subset:
                    pattern_var_name = f"P_{abs_pos}"
                    if pattern_var_name in result["solved_patterns"]:
                        pattern_value = result["solved_patterns"][pattern_var_name]
                        patch_patterns.append(pattern_value)

                cat_patches[cat_name].append({"subset": subset, "weight": weight, "patterns": patch_patterns})

            for cat_name in sorted(cat_patches.keys()):
                patches = cat_patches[cat_name]
                total_weight = sum(patch["weight"] for patch in patches)
                print(f"  {cat_name}: {len(patches)} patches, weight {total_weight}")
                for i, patch in enumerate(patches, 1):
                    subset_str = ", ".join(map(str, sorted(patch["subset"])))
                    patterns_str = ", ".join(map(str, patch["patterns"]))
                    print(f"    Patch {i}: [{subset_str}] (weight {patch['weight']}, patterns: [{patterns_str}])")

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
                tile_positions_str = ", ".join(f"({pos.q},{pos.r})" for pos in tile_positions)
                abs_positions_str = ", ".join(str(pos.abs) for pos in tile_positions)
                print(f"  Component {idx}: Subset {subset_idx} with {color_name}")
                print(f"    Tiles: {tile_positions_str} [Abs: {abs_positions_str}]")

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


def save_combined_result_to_log(combined_components: CombinedModelComponents, result: dict) -> str:
    """
    Save the combined result to a log file.

    Args:
        combined_components: Combined model components
        result: Solution result dictionary

    Returns:
        Filename of the created log file
    """
    # Create filename
    board_name = result["board_setting"]
    design_goal_str = "_".join(result["design_goal_names"])
    cats_str = "_".join(result["cat_names"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/COMBINED_{board_name}_{design_goal_str}_{cats_str}_{timestamp}.log"

    with open(filename, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("CALICO COMBINED SOLVER RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Board: {result['board_setting']}\n")
        f.write(f"Design Goals: {' -> '.join(result['design_goal_names'])}\n")
        f.write(f"Cats: {', '.join(result['cat_names'])}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write solver results
        f.write(f"Status: {result['status']}\n")

        if result.get("objective_value") is not None:
            f.write(f"Total Objective Value: {result['objective_value_mod']})\n")
            f.write(f"Solve Time: {result.get('solve_time', 'N/A')}s\n\n")

            # Write detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 60 + "\n")

            # Design Goals section
            f.write("DESIGN GOALS RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Design Goals Score: {result['total_design_goals_score']}\n")
            for i, (_goal_name, score) in enumerate(result["design_goal_scores"].items()):
                goal_tile = combined_components.design_goals_model.design_goal_tiles[i]
                if score == goal_tile.higher_points:
                    status = "FULL (Pattern + Color)"
                elif score == goal_tile.lower_points:
                    status = "PARTIAL (Pattern or Color)"
                else:
                    status = "NONE"
                f.write(
                    f"  Goal {i + 1} ({goal_tile.config_name}): {score}/{goal_tile.higher_points} points - {status}\n"
                )
            f.write("\n")

            # Cats section
            f.write("CATS RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cats Objective: {result['cats_objective_value']}\n")
            f.write(f"Patches Selected: {result['cats_count']}\n")

            if result["cats_chosen_indices"]:
                # Group by cat
                cat_patches = {}
                for idx in result["cats_chosen_indices"]:
                    weight = combined_components.cats_components.weights[idx]
                    cat_name = combined_components.cats_components.cat_for_weight.get(
                        weight, f"Unknown (weight {weight})"
                    )

                    if cat_name not in cat_patches:
                        cat_patches[cat_name] = []

                    subset = combined_components.cats_components.subsets[idx]

                    # Get patterns for this patch
                    patch_patterns = []
                    for abs_pos in subset:
                        pattern_var_name = f"P_{abs_pos}"
                        if pattern_var_name in result["solved_patterns"]:
                            pattern_value = result["solved_patterns"][pattern_var_name]
                            patch_patterns.append(pattern_value)

                    cat_patches[cat_name].append({"subset": subset, "weight": weight, "patterns": patch_patterns})

                for cat_name in sorted(cat_patches.keys()):
                    patches = cat_patches[cat_name]
                    total_weight = sum(patch["weight"] for patch in patches)
                    f.write(f"  {cat_name}: {len(patches)} patches, weight {total_weight}\n")
                    for i, patch in enumerate(patches, 1):
                        subset_str = ", ".join(map(str, sorted(patch["subset"])))
                        patterns_str = ", ".join(map(str, patch["patterns"]))
                        f.write(
                            f"    Patch {i}: [{subset_str}] (weight {patch['weight']}, patterns: [{patterns_str}])\n"
                        )
            f.write("\n")

            # Buttons section
            f.write("BUTTONS RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Buttons Objective: {result['buttons_objective_value']}\n")
            f.write(f"K-consistent Components: {len(result['buttons_representatives'])}\n")
            f.write(f"Rainbow Bonus: {'Yes' if result['buttons_bonus'] else 'No'}\n")

            if result["buttons_representatives"]:
                from core.enums.color import Color

                color_names = list(Color.__members__.keys())

                f.write("Representative Components:\n")
                for idx, (subset_idx, color_idx) in enumerate(result["buttons_representatives"], 1):
                    tile_positions = combined_components.buttons_components.tile_sets[subset_idx]
                    color_name = color_names[color_idx]
                    tile_positions_str = ", ".join(f"({pos.q},{pos.r})" for pos in tile_positions)
                    abs_positions_str = ", ".join(str(pos.abs) for pos in tile_positions)
                    f.write(f"  Component {idx}: Subset {subset_idx} with {color_name}\n")
                    f.write(f"    Tiles: {tile_positions_str} [Abs: {abs_positions_str}]\n")
            f.write("\n")

            # Board solution
            f.write("QUILT BOARD SOLUTION:\n")
            f.write("-" * 60 + "\n")
            solution_board = create_solution_board(combined_components, result)
            f.write(solution_board.pretty_print() + "\n\n")

        else:
            f.write("No solution found.\n\n")

        f.write("=" * 80 + "\n")

    return filename


def main():
    """Example run of the combined solver."""
    print("Running combined solver example...")

    # Configuration
    design_goals = (
        DesignGoalTiles.FOUR_TWO.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
        DesignGoalTiles.THREE_PAIRS.value,
    )
    cat_names = ("leo", "cira", "rumi")  # One from each difficulty group
    board_setting = EdgeTileSettings.BOARD_1

    # Build model
    combined_components = build_combined_model(
        design_goal_tiles=design_goals,
        cat_names=cat_names,
        board_setting=board_setting,
        design_goals_weight=1.0,
        cats_weight=1.0,
        buttons_weight=3.0,
        time_limit_s=300.0,
    )

    # Solve model
    result = solve_combined_model(combined_components, time_limit_sec=300.0, workers=8)

    # Print results
    print_combined_solution(combined_components, result)

    # Save to log
    if result.get("objective_value") is not None:
        log_filename = save_combined_result_to_log(combined_components, result)
        print(f"\nResults saved to: {log_filename}")


if __name__ == "__main__":
    main()
