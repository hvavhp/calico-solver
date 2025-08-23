from ortools.sat.python import cp_model

from core.enums.color import COLOR_MAP, Color
from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import Pattern
from core.models.patch_tile import PatchTile
from core.models.quilt_board import HexPosition, QuiltBoard
from solvers.restructured_design_goals_solver import DesignGoalsModel
from solvers.restructured_design_goals_solver import build_model as design_goals_build_model


def build_model(base_model: DesignGoalsModel | None = None) -> DesignGoalsModel:
    """
    n: number of base vars x[0..n-1]
    values: [v1,..,vK] (K can be 6 as in your case)
    m: list of subset specs over indices of x; each item either:
       - {"first":[...], "second":[...]}  OR
       - {"subset":[...], "t": t} meaning first=subset[:t], second=subset[t:].
    edges: list of undirected edges on subset indices, e.g. [(0,1),(1,2),...]
    add_additional_constraints: optional callback(model, x, values) to add your original constraints on x.
    """
    quilt_board = QuiltBoard(
        edge_setting=EdgeTileSettings.BOARD_1,
        design_goal_tiles=[
            DesignGoalTiles.FOUR_TWO.value,
            DesignGoalTiles.THREE_TWO_ONE.value,
            DesignGoalTiles.THREE_PAIRS.value,
        ],
    )

    model = base_model.model

    # model = cp_model.CpModel()
    colors = COLOR_MAP

    # 1) Decision vars: x_i in allowed set
    # x = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(values), f"x[{i}]") for i in range(n)]
    x = base_model.color_variables

    # 2) Literals eq[(i,k)] <=> (x[i] == values[k])
    eq = base_model.z_color_variables

    # get 3-tile sets and 2-tile sets near edge
    three_tile_sets = quilt_board.get_three_neighbor_tile_sets()
    two_tile_sets = quilt_board.get_two_neighbor_tile_sets_near_edge()
    tile_sets = three_tile_sets + two_tile_sets
    s = len(tile_sets)

    # build set graph edges
    set_graph_edges = []
    for s_idx, _s in enumerate(tile_sets):
        for t_idx, _t in enumerate(tile_sets):
            # check if the two sets share any tiles
            if any(tile in _t for tile in _s):
                set_graph_edges.append((s_idx, t_idx))
                continue

            s_neighbors = [quilt_board._get_hex_neighbors(tile) for tile in _s]
            s_neighbors = [neighbor for neighbor in s_neighbors if neighbor in _t]
            s_neighbors = list(set(s_neighbors))

            if any(neighbor in _t for neighbor in s_neighbors):
                set_graph_edges.append((s_idx, t_idx))

    # 4) Compute y_{S,k} and y_S
    y_sk = [[None for _ in range(len(colors))] for _ in range(s)]
    y_s = [model.NewBoolVar(f"y_subset[{_s}]") for _s in range(s)]

    r = {}
    for _s, subset in enumerate(tile_sets):
        # One literal per k saying subset s wins with value v_k
        sum_k = []
        _colors = list(colors.keys())
        if len(subset) == 2:
            _colors = [quilt_board.get_edge_colors_of_tile(tile) for tile in subset]
            _colors = [color for _l in _colors for color in _l]
            _colors = list(set(_colors))

        for _color in colors:
            _k = colors[_color]
            conds = [eq[tile.abs][_k] for tile in subset]
            y_sk[_s][_k] = model.NewBoolVar(f"y_subset[{_s}]_val{_k}")

            if _color not in _colors:
                model.Add(y_sk[_s][_k] == 0)
                continue

            # ySk[s][k] <=> AND(conds)
            for lit in conds:
                model.AddImplication(y_sk[_s][_k], lit)
            model.AddBoolOr([c.Not() for c in conds] + [y_sk[_s][_k]])  # AND => ySk
            model.AddBoolAnd(conds).OnlyEnforceIf(y_sk[_s][_k])
            sum_k.append(y_sk[_s][_k])

        # With non-empty first group, at most one k can be true; tie equality for convenience:
        model.Add(sum(sum_k) == y_s[_s])

    # 5) k-consistent component counting via representatives
    # r[R][k] == 1  <=> R is representative of a k-labeled component
    r = [[model.NewBoolVar(f"r[{_r}][{_k}]") for _k in range(len(colors))] for _r in range(s)]
    # a[S][R][k] == 1  <=> S belongs to representative R in label k
    a = [
        [[model.NewBoolVar(f"a[{_s}][{_r}][{_k}]") for _k in range(len(colors))] for _r in range(s)] for _s in range(s)
    ]

    # Every winning vertex (for k) picks exactly one representative (for k)
    for _s in range(s):
        for _color in range(len(colors)):
            model.Add(sum(a[_s][_r][_color] for _r in range(s)) == y_sk[_s][_color])

    # Representative consistency and self-labeling
    for _r in range(s):
        for _color in range(len(colors)):
            model.Add(a[_r][_r][_color] == r[_r][_color])
            for _s in range(s):
                model.Add(a[_s][_r][_color] <= r[_r][_color])

    # Edge-induced equality (only when both endpoints win with the same k)
    for u, v in set_graph_edges:
        for _r in range(s):
            for _color in range(len(colors)):
                model.Add(a[u][_r][_color] == a[v][_r][_color]).OnlyEnforceIf([y_sk[u][_color], y_sk[v][_color]])

    # Objective: total number of k-consistent activated components
    model.Maximize(sum(r[_r][_k] for _r in range(s) for _k in range(len(colors))))

    return model, x, y_s, y_sk, r, a


def solve_model(model, base_model: DesignGoalsModel, x, y_s, y_sk, r, a, time_limit_sec=None, workers=8):
    solver = cp_model.CpSolver()
    if time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = workers

    status = solver.Solve(model)
    out = {"status": solver.StatusName(status)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["objective"] = solver.ObjectiveValue()
        out["x"] = [solver.Value(v) for v in x.values()]
        out["yS"] = [solver.Value(v) for v in y_s]

        # Extract solved colors and patterns from base_model
        solved_colors = {}
        solved_patterns = {}

        # Get color solutions
        for var_name, var in base_model.color_variables.items():
            solved_colors[var_name] = solver.Value(var)

        # Get pattern solutions
        for var_name, var in base_model.pattern_variables.items():
            solved_patterns[var_name] = solver.Value(var)

        out["solved_colors"] = solved_colors
        out["solved_patterns"] = solved_patterns

        # Decode (k, representative) for each winning subset
        comps = []
        for _s in range(len(y_s)):
            if solver.Value(y_s[_s]) == 0:
                comps.append(None)
                continue
            # find its k
            k_star = None
            for _k in range(len(y_sk[_s])):
                if solver.Value(y_sk[_s][_k]) == 1:
                    k_star = _k
                    break
            # find its representative R under k_star
            r_star = None
            for _r in range(len(r)):
                if solver.Value(a[_s][_r][k_star]) == 1:
                    r_star = _r
                    break
            comps.append((k_star, r_star))
        out["component_of_subset"] = comps

        # Add representative information
        representatives = []
        for _r in range(len(r)):
            for _k in range(len(r[_r])):
                if solver.Value(r[_r][_k]) == 1:
                    representatives.append((_r, _k))
        out["representatives"] = representatives
    return out


def create_solution_board(base_board: QuiltBoard, result: dict, patch_tiles: list[HexPosition]) -> QuiltBoard:
    """Create a QuiltBoard showing the solved colors and patterns.

    Args:
        base_board: The base QuiltBoard with edge tiles and design goals
        result: The solver result dictionary containing solved colors and patterns
        patch_tiles: List of HexPosition objects for patch tiles

    Returns:
        QuiltBoard with solved colors and patterns filled in
    """
    # Create a new board with the same configuration
    solution_board = QuiltBoard(edge_setting=base_board.edge_setting, design_goal_tiles=base_board.design_goal_tiles)

    # Map color and pattern values back to enums
    colors = list(COLOR_MAP.keys())
    patterns = list(Pattern)

    # Get solved colors and patterns
    solved_colors = result.get("solved_colors", {})
    solved_patterns = result.get("solved_patterns", {})

    # Fill in the patch tiles with solved colors and patterns
    for patch_pos in patch_tiles:
        abs_pos = patch_pos.abs
        color_var_name = f"C_{abs_pos}"
        pattern_var_name = f"P_{abs_pos}"

        # Get solved color
        if color_var_name in solved_colors:
            color_value = solved_colors[color_var_name]
            solved_color = colors[color_value]
        else:
            # Fallback to arbitrary color if not found
            solved_color = colors[0]

        # Get solved pattern
        if pattern_var_name in solved_patterns:
            pattern_value = solved_patterns[pattern_var_name]
            solved_pattern = patterns[pattern_value]
        else:
            # Fallback to arbitrary pattern if not found
            solved_pattern = patterns[0]

        # Create the patch tile
        patch_tile = PatchTile(color=solved_color, pattern=solved_pattern)

        # Place it on the board
        solution_board.tiles_by_pos[patch_pos] = patch_tile

    return solution_board


def display_scoring_details(board: QuiltBoard, result: dict, patch_tiles: list[HexPosition]) -> None:
    """Display detailed scoring information for the solution.

    Args:
        board: The QuiltBoard used for the model
        result: The solver result dictionary
        patch_tiles: List of HexPosition objects for patch tiles
    """
    colors = list(COLOR_MAP.keys())
    color_names = list(Color.__members__.keys())

    # Get the tile sets used in the model
    three_tile_sets = board.get_three_neighbor_tile_sets()
    two_tile_sets = board.get_two_neighbor_tile_sets_near_edge()
    all_tile_sets = three_tile_sets + two_tile_sets

    # Get solved colors and patterns
    solved_colors = result.get("solved_colors", {})
    solved_patterns = result.get("solved_patterns", {})

    # Create mapping from patch tile to its solved color and pattern
    tile_colors = {}
    tile_patterns = {}
    patterns = list(Pattern)

    for patch_pos in patch_tiles:
        abs_pos = patch_pos.abs
        color_var_name = f"C_{abs_pos}"
        pattern_var_name = f"P_{abs_pos}"

        # Get solved color
        if color_var_name in solved_colors:
            color_value = solved_colors[color_var_name]
            solved_color = colors[color_value]
            tile_colors[patch_pos] = solved_color

        # Get solved pattern
        if pattern_var_name in solved_patterns:
            pattern_value = solved_patterns[pattern_var_name]
            solved_pattern = patterns[pattern_value]
            tile_patterns[patch_pos] = solved_pattern

    # Show maximum score
    max_score = result["objective"]
    print(f"Maximum Score Achieved: {max_score}")
    print(f"This represents {int(max_score)} k-consistent activated components")
    print()

    # Only show representative subsets where r[_r][_color] = 1
    representatives = result.get("representatives", [])

    if representatives:
        print(f"Representative Components ({len(representatives)} total):")
        print("-" * 50)

        for idx, (subset_idx, color_idx) in enumerate(representatives, 1):
            tile_positions = all_tile_sets[subset_idx]
            color_name = color_names[color_idx]

            print(f"Component {idx}: Subset {subset_idx} with {color_name}")

            # Show tile positions with their colors and patterns
            tile_details = []
            for pos in tile_positions:
                color = tile_colors.get(pos)
                pattern = tile_patterns.get(pos)
                if color and pattern:
                    actual_color_name = color_names[colors.index(color)]
                    pattern_name = pattern.value.title()
                    tile_details.append(f"({pos.q},{pos.r}):{actual_color_name}-{pattern_name}")
                elif color:
                    actual_color_name = color_names[colors.index(color)]
                    tile_details.append(f"({pos.q},{pos.r}):{actual_color_name}")

            print(f"  Tiles: {', '.join(tile_details)}")
            print()
    else:
        print("No representative components found.")
        print()

    # Show color and pattern distribution
    print("Solution Summary:")
    print("-" * 30)

    # Color distribution
    color_counts = {}
    for color in tile_colors.values():
        color_counts[color] = color_counts.get(color, 0) + 1

    print("Color Distribution:")
    for color in colors:
        count = color_counts.get(color, 0)
        color_name = color_names[colors.index(color)]
        print(f"  {color_name}: {count} tiles")

    print()

    # Pattern distribution
    pattern_counts = {}
    for pattern in tile_patterns.values():
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    print("Pattern Distribution:")
    for pattern in patterns:
        count = pattern_counts.get(pattern, 0)
        pattern_name = pattern.value.title()
        print(f"  {pattern_name}: {count} tiles")


def main():
    model = cp_model.CpModel()
    v = [COLOR_MAP[color] for color in Color]

    m1 = DesignGoalTiles.FOUR_TWO.value
    m2 = DesignGoalTiles.SIX_UNIQUE.value
    m3 = DesignGoalTiles.TWO_TRIPLETS.value

    board = QuiltBoard(
        design_goal_tiles=[
            m1,
            m2,
            m3,
        ]
    )

    base_model = design_goals_build_model(model, v, m1, m2, m3, cap=3, time_limit_s=200)

    # patch_tiles = board.get_all_patch_tiles()

    # variable_indices = [i.abs for i in patch_tiles]
    # color_variable_names = [f"C_{i.abs}" for i in patch_tiles]

    # color_variables = {name: model.NewIntVarFromDomain(dom, name) for name in color_variable_names}
    # b_colors = add_channeling(model, list(color_variables.values()), variable_indices, v, "C")
    # b_color_map = dict(zip(variable_indices, b_colors, strict=False))

    # base_model = DesignGoalsModel(
    #     model=model,
    #     pattern_variables={},
    #     color_variables=color_variables,
    #     z_pattern_variables={},
    #     z_color_variables=b_color_map,
    #     pair_indicators=[],
    #     variable_indices=variable_indices,
    #     k=len(v),
    #     design_goal_tiles=board.design_goal_tiles,
    #     cap=3,
    #     time_limit_s=200,
    # )

    model, x, y_s, y_sk, r, a = build_model(base_model)
    res = solve_model(model, base_model, x, y_s, y_sk, r, a, time_limit_sec=200)

    print("Status:", res["status"])
    if "objective" in res:
        print("Objective (k-consistent activated components):", res["objective"])
        print("x:", res["x"])
        print("winners yS:", res["yS"])
        print("component (k, representative) per winning subset:", res["component_of_subset"])

        # Pretty print the solution board
        print("\n" + "=" * 60)
        print("QUILT BOARD SOLUTION")
        print("=" * 60)

        # Create a solution board with the solved colors and patterns
        solution_board = create_solution_board(board, res, board.get_all_patch_tiles())
        print(solution_board.pretty_print())

        # Display scoring details
        print("\n" + "=" * 60)
        print("SCORING DETAILS")
        print("=" * 60)
        display_scoring_details(board, res, board.get_all_patch_tiles())


if __name__ == "__main__":
    main()
