# objective_design_goals_solver.py
# A version of the design goals solver that treats design goals as optimization objectives
# rather than hard constraints. This allows for partial satisfaction and flexible scoring.

from ortools.sat.python import cp_model
from pydantic import BaseModel

from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import PATTERN_MAP
from core.models.design_goal_tile import DesignGoalTile
from core.models.quilt_board import QuiltBoard


def add_channeling(
    model: cp_model.CpModel, vars: list[cp_model.IntVar], var_indices: list[int], values: list[int], tag: str
) -> list[list[cp_model.IntVar]]:
    """Create channeling booleans b[i][k] <-> (g[i] == values[k]), with exactly-one per i."""
    n, k = len(vars), len(values)
    b = [[model.NewBoolVar(f"b[{tag}][i={i},k={k_val}]") for k_val in range(k)] for i in var_indices]
    for i in range(n):
        # model.Add(sum(b[i][k_val] for k_val in range(k)) == 1)
        for k_val in range(k):
            model.Add(vars[i] == values[k_val]).OnlyEnforceIf(b[i][k_val])
            model.Add(vars[i] != values[k_val]).OnlyEnforceIf(b[i][k_val].Not())
    return b


def add_soft_pattern_scoring(
    model: cp_model.CpModel, b: list[list[cp_model.IntVar]], required_mults: list[int], tag: str
) -> cp_model.IntVar:
    """
    Add soft scoring for pattern matching instead of hard constraints.
    Returns a boolean variable that is True if the pattern is satisfied.

    Args:
        model: The CP-SAT model
        b: Boolean variables indicating which value is assigned to each position
        required_mults: Required multiplicities (e.g., [2, 2, 2] for three pairs)
        tag: String tag for variable naming

    Returns:
        Boolean variable that is True if the pattern matches the requirement
    """
    n, k = len(b), len(b[0])
    r = len(required_mults)

    # Create assignment variables: w[k_val][slot] = 1 if value k_val is assigned to slot 'slot'
    w = [[model.NewBoolVar(f"w[{tag}][k={k_val},s={s}]") for s in range(r)] for k_val in range(k)]

    # Each slot must be taken by exactly one value
    for s in range(r):
        model.Add(sum(w[k_val][s] for k_val in range(k)) == 1)

    # Each value can serve at most one slot
    for k_val in range(k):
        model.Add(sum(w[k_val][s] for s in range(r)) <= 1)

    # Create matched variables for each value - much simpler approach
    matched_k = []
    for k_val in range(k):
        # Count how many times this value appears
        occ_k = sum(b[i][k_val] for i in range(n))

        # Create boolean: matched_k[k_val] = True iff value k_val satisfies its count requirement
        matched = model.NewBoolVar(f"matched[{tag}][k={k_val}]")
        matched_k.append(matched)

        # Only enforce the counting constraint if matched = True
        model.Add(occ_k == sum(required_mults[s] * w[k_val][s] for s in range(r))).OnlyEnforceIf(matched)
        model.Add(occ_k != sum(required_mults[s] * w[k_val][s] for s in range(r))).OnlyEnforceIf(matched.Not())

    # Pattern is satisfied if all values are matched
    pattern_satisfied = model.NewBoolVar(f"pattern_satisfied[{tag}]")
    model.AddBoolAnd(matched_k).OnlyEnforceIf(pattern_satisfied)
    model.AddBoolOr([var.Not() for var in matched_k]).OnlyEnforceIf(pattern_satisfied.Not())

    # Add symmetry breaking for equal multiplicities (optional speedup)
    eq = {}
    for s, sz in enumerate(required_mults):
        eq.setdefault(sz, []).append(s)
    for slots in eq.values():
        for a, bslot in zip(slots, slots[1:], strict=False):
            model.Add(
                sum((k_val + 1) * w[k_val][a] for k_val in range(k))
                <= sum((k_val + 1) * w[k_val][bslot] for k_val in range(k))
            )

    return pattern_satisfied


def add_pair_indicators(
    model: cp_model.CpModel, b_p: list[list[cp_model.IntVar]], b_c: list[list[cp_model.IntVar]], tag: str
) -> list[list[list[cp_model.IntVar]]]:
    """Create pair[i][k][l] = AND(bU[i][k], bP[i][l]); also exactly-one per i."""
    n, k = len(b_p), len(b_p[0])
    pair = [
        [[model.NewBoolVar(f"pair[{tag}][i={i},k={_k},l={_l}]") for _l in range(k)] for _k in range(k)]
        for i in range(n)
    ]
    for i in range(n):
        model.Add(sum(pair[i][_k][_l] for _k in range(k) for _l in range(k)) == 1)
        for _k in range(k):
            for _l in range(k):
                # linear AND
                model.Add(pair[i][_k][_l] <= b_p[i][_k])
                model.Add(pair[i][_k][_l] <= b_c[i][_l])
                model.Add(pair[i][_k][_l] >= b_p[i][_k] + b_c[i][_l] - 1)
    return pair


def create_quilt_board_from_solution(pattern_values, color_values, variable_indices, design_goal_tiles):
    """Create a quilt board from the solution's pattern and color values."""
    from core.enums.color import ALL_COLORS
    from core.enums.pattern import ALL_PATTERNS
    from core.models.patch_tile import PatchTile
    from core.models.quilt_board import HexPosition, QuiltBoard

    # Create mapping from numeric values [1,2,3,4,5,6] to actual enums
    value_to_color = {i + 1: ALL_COLORS[i] for i in range(6)}
    value_to_pattern = {i + 1: ALL_PATTERNS[i] for i in range(6)}

    # Create the board with design goal tiles
    board = QuiltBoard(design_goal_tiles=design_goal_tiles)

    # Add patch tiles based on solution values
    for _, abs_pos in enumerate(variable_indices):
        # Convert absolute position back to hex coordinates
        q = abs_pos % 7
        r = abs_pos // 7
        hex_pos = HexPosition(q=q, r=r)

        # Get pattern and color values for this position
        pattern_var_name = f"P_{abs_pos}"
        color_var_name = f"C_{abs_pos}"

        if pattern_var_name in pattern_values and color_var_name in color_values:
            pattern_value = pattern_values[pattern_var_name]
            color_value = color_values[color_var_name]

            # Create patch tile with the mapped color and pattern
            patch_tile = PatchTile(color=value_to_color[color_value], pattern=value_to_pattern[pattern_value])

            # Place the patch tile on the board
            board.tiles_by_pos[hex_pos] = patch_tile

    return board


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Callback to print and collect all solutions with design goal scoring."""

    def __init__(
        self,
        pattern_variables,
        color_variables,
        pair_indicators,
        variable_indices,
        k,
        design_goal_tiles,
        design_goal_scores,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._pattern_variables = pattern_variables
        self._color_variables = color_variables
        self._pair_indicators = pair_indicators
        self._variable_indices = variable_indices
        self._k = k
        self._design_goal_tiles = design_goal_tiles
        self._design_goal_scores = design_goal_scores
        self._solutions = []
        self._solution_count = 0

    def _create_quilt_board_from_solution(self, pattern_values, color_values):
        """Create a quilt board from the solution's pattern and color values."""
        from core.enums.color import ALL_COLORS
        from core.enums.pattern import ALL_PATTERNS
        from core.models.patch_tile import PatchTile
        from core.models.quilt_board import HexPosition, QuiltBoard

        # Create mapping from numeric values [1,2,3,4,5,6] to actual enums
        value_to_color = {i + 1: ALL_COLORS[i] for i in range(6)}
        value_to_pattern = {i + 1: ALL_PATTERNS[i] for i in range(6)}

        # Create the board with design goal tiles
        board = QuiltBoard(design_goal_tiles=self._design_goal_tiles)

        # Add patch tiles based on solution values
        for _, abs_pos in enumerate(self._variable_indices):
            # Convert absolute position back to hex coordinates
            q = abs_pos % 7
            r = abs_pos // 7
            hex_pos = HexPosition(q=q, r=r)

            # Get pattern and color values for this position
            pattern_var_name = f"P_{abs_pos}"
            color_var_name = f"C_{abs_pos}"

            if pattern_var_name in pattern_values and color_var_name in color_values:
                pattern_value = pattern_values[pattern_var_name]
                color_value = color_values[color_var_name]

                # Create patch tile with the mapped color and pattern
                patch_tile = PatchTile(color=value_to_color[color_value], pattern=value_to_pattern[pattern_value])

                # Place the patch tile on the board
                board.tiles_by_pos[hex_pos] = patch_tile

        return board

    def on_solution_callback(self):
        self._solution_count += 1
        print(f"\n--- Solution {self._solution_count} ---")

        # Extract variable values
        sol = {}

        # Extract pattern variable values
        pattern_values = {}
        for name, var in self._pattern_variables.items():
            pattern_values[name] = self.Value(var)
        sol["patterns"] = pattern_values
        print("Pattern variables:", pattern_values)

        # Extract color variable values
        color_values = {}
        for name, var in self._color_variables.items():
            color_values[name] = self.Value(var)
        sol["colors"] = color_values
        print("Color variables:", color_values)

        # Extract design goal scores
        design_goal_score_values = {}
        total_design_goal_score = 0
        for name, var in self._design_goal_scores.items():
            score = self.Value(var)
            design_goal_score_values[name] = score
            total_design_goal_score += score
        sol["design_goal_scores"] = design_goal_score_values
        print("Design Goal Scores:", design_goal_score_values)
        print("Total Design Goal Score:", total_design_goal_score)

        # Create and pretty print the quilt board
        board = self._create_quilt_board_from_solution(pattern_values, color_values)
        print("\nQuilt Board:")
        print(board.pretty_print())

        # Extract pair counts from pair indicators
        pair_counts = [[0] * self._k for _ in range(self._k)]
        for i in range(len(self._pair_indicators)):
            for k_val in range(self._k):
                for val in range(self._k):
                    if self.Value(self._pair_indicators[i][k_val][val]):
                        pair_counts[k_val][val] += 1

        sol["pair_counts_total"] = pair_counts
        print("Pair counts across positions (k rows, val cols):")
        for row in pair_counts:
            print(row)

        self._solutions.append(sol)

    def get_solutions(self):
        return self._solutions

    def get_solution_count(self):
        return self._solution_count


class ObjectiveDesignGoalsModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: cp_model.CpModel
    pattern_variables: dict[str, cp_model.IntVar]
    color_variables: dict[str, cp_model.IntVar]
    z_pattern_variables: dict[int, list[cp_model.IntVar]]
    z_color_variables: dict[int, list[cp_model.IntVar]]
    pair_indicators: list[list[list[cp_model.IntVar]]]
    variable_indices: list[int]
    k: int
    design_goal_tiles: list[DesignGoalTile]
    design_goal_scores: dict[str, cp_model.IntVar]
    cap: int
    time_limit_s: float


def solve_objective_design_goals(
    v: list[int], m1: DesignGoalTile, m2: DesignGoalTile, m3: DesignGoalTile, cap: int = 3, time_limit_s: float = 5.0
):
    """Solve design goals as optimization objectives rather than hard constraints."""
    design_goals_model = build_objective_model(None, v, m1, m2, m3, cap, time_limit_s)

    # Set the objective function to maximize total design goal score
    total_score = sum(design_goals_model.design_goal_scores.values())
    design_goals_model.model.Maximize(total_score)

    # Search for optimal solution
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s

    print("Searching for optimal solution...")
    status = solver.Solve(design_goals_model.model)

    print(f"\nSearch completed with status: {solver.StatusName(status)}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Extract the solution
    pattern_values = {}
    for name, var in design_goals_model.pattern_variables.items():
        pattern_values[name] = solver.Value(var)

    color_values = {}
    for name, var in design_goals_model.color_variables.items():
        color_values[name] = solver.Value(var)

    # Extract design goal scores
    design_goal_score_values = {}
    total_design_goal_score = 0
    for name, var in design_goals_model.design_goal_scores.items():
        score = solver.Value(var)
        design_goal_score_values[name] = score
        total_design_goal_score += score

    print("Pattern variables:", pattern_values)
    print("Color variables:", color_values)
    print("Design Goal Scores:", design_goal_score_values)
    print("Total Design Goal Score:", total_design_goal_score)

    # Create and pretty print the quilt board
    board = create_quilt_board_from_solution(
        pattern_values, color_values, design_goals_model.variable_indices, design_goals_model.design_goal_tiles
    )
    print("\nQuilt Board:")
    print(board.pretty_print())

    # Extract pair counts from pair indicators
    pair_counts = [[0] * design_goals_model.k for _ in range(design_goals_model.k)]
    for i in range(len(design_goals_model.pair_indicators)):
        for k_val in range(design_goals_model.k):
            for val in range(design_goals_model.k):
                if solver.Value(design_goals_model.pair_indicators[i][k_val][val]):
                    pair_counts[k_val][val] += 1

    print("Pair counts across positions (k rows, val cols):")
    for row in pair_counts:
        print(row)

    return {
        "patterns": pattern_values,
        "colors": color_values,
        "design_goal_scores": design_goal_score_values,
        "total_score": total_design_goal_score,
        "pair_counts_total": pair_counts,
        "board": board,
        "status": solver.StatusName(status),
    }


def build_objective_model(
    model: cp_model.CpModel | None = None,
    v: list[int] | None = None,
    m1: DesignGoalTile = DesignGoalTiles.THREE_PAIRS.value,
    m2: DesignGoalTile = DesignGoalTiles.THREE_TWO_ONE.value,
    m3: DesignGoalTile = DesignGoalTiles.FOUR_TWO.value,
    cap: int = 3,
    time_limit_s: float = 5.0,
    edge_setting: EdgeTileSettings = EdgeTileSettings.BOARD_1,
) -> ObjectiveDesignGoalsModel:
    """
    Build a model where design goals are treated as optimization objectives.

    Args:
        model: Optional existing CP model
        v: List of 6 distinct ints representing values
        m1: First design goal tile with configuration
        m2: Second design goal tile with configuration
        m3: Third design goal tile with configuration
        cap: Max allowed count for any ordered pair across positions
        time_limit_s: Time limit for solving
        edge_setting: Board edge configuration

    Returns:
        ObjectiveDesignGoalsModel containing the model and variables
    """
    if v is None:
        v = list(PATTERN_MAP.values())
    assert len(v) == 6
    k = 6

    if not model:
        model = cp_model.CpModel()

    dom = cp_model.Domain.FromValues(v)

    board = QuiltBoard(edge_setting=edge_setting, design_goal_tiles=[m1, m2, m3])
    patch_tiles = board.get_all_patch_tiles()

    variable_indices = [i.abs for i in patch_tiles]
    pattern_variable_names = [f"P_{i.abs}" for i in patch_tiles]
    color_variable_names = [f"C_{i.abs}" for i in patch_tiles]

    pattern_variables = {name: model.NewIntVarFromDomain(dom, name) for name in pattern_variable_names}
    color_variables = {name: model.NewIntVarFromDomain(dom, name) for name in color_variable_names}

    b_patterns = add_channeling(model, list(pattern_variables.values()), variable_indices, v, "P")
    b_pattern_map = dict(zip(variable_indices, b_patterns, strict=False))
    b_colors = add_channeling(model, list(color_variables.values()), variable_indices, v, "C")
    b_color_map = dict(zip(variable_indices, b_colors, strict=False))

    # Get design goal patch tiles for each goal
    design_goal_patch_tiles_0 = board.get_design_goal_patch_tiles(0)
    variable_indices_0 = [i.abs for i in design_goal_patch_tiles_0]
    b_patterns_0 = [b_pattern_map[i] for i in variable_indices_0]
    b_colors_0 = [b_color_map[i] for i in variable_indices_0]

    design_goal_patch_tiles_1 = board.get_design_goal_patch_tiles(1)
    variable_indices_1 = [i.abs for i in design_goal_patch_tiles_1]
    b_patterns_1 = [b_pattern_map[i] for i in variable_indices_1]
    b_colors_1 = [b_color_map[i] for i in variable_indices_1]

    design_goal_patch_tiles_2 = board.get_design_goal_patch_tiles(2)
    variable_indices_2 = [i.abs for i in design_goal_patch_tiles_2]
    b_patterns_2 = [b_pattern_map[i] for i in variable_indices_2]
    b_colors_2 = [b_color_map[i] for i in variable_indices_2]

    # Create soft scoring variables for each design goal
    design_goal_scores = {}

    # Goal 0: Pattern and Color satisfaction
    pattern_sat_0 = add_soft_pattern_scoring(model, b_patterns_0, m1.config_numbers, "goal0_P")
    color_sat_0 = add_soft_pattern_scoring(model, b_colors_0, m1.config_numbers, "goal0_C")

    # Calculate score for goal 0
    both_sat_0 = model.NewBoolVar("both_satisfied_goal0")
    model.AddBoolAnd([pattern_sat_0, color_sat_0]).OnlyEnforceIf(both_sat_0)
    model.AddBoolOr([pattern_sat_0.Not(), color_sat_0.Not()]).OnlyEnforceIf(both_sat_0.Not())

    either_sat_0 = model.NewBoolVar("either_satisfied_goal0")
    model.AddBoolOr([pattern_sat_0, color_sat_0]).OnlyEnforceIf(either_sat_0)
    model.AddBoolAnd([pattern_sat_0.Not(), color_sat_0.Not()]).OnlyEnforceIf(either_sat_0.Not())

    partial_only_0 = model.NewBoolVar("partial_only_goal0")
    model.AddBoolAnd([either_sat_0, both_sat_0.Not()]).OnlyEnforceIf(partial_only_0)
    model.AddBoolOr([either_sat_0.Not(), both_sat_0]).OnlyEnforceIf(partial_only_0.Not())

    score_0 = model.NewIntVar(0, m1.higher_points, "score_goal0")
    model.Add(score_0 == m1.higher_points).OnlyEnforceIf(both_sat_0)
    model.Add(score_0 == m1.lower_points).OnlyEnforceIf(partial_only_0)
    model.Add(score_0 == 0).OnlyEnforceIf([both_sat_0.Not(), partial_only_0.Not()])

    design_goal_scores["goal_0"] = score_0

    # Goal 1: Similar logic
    pattern_sat_1 = add_soft_pattern_scoring(model, b_patterns_1, m2.config_numbers, "goal1_P")
    color_sat_1 = add_soft_pattern_scoring(model, b_colors_1, m2.config_numbers, "goal1_C")

    both_sat_1 = model.NewBoolVar("both_satisfied_goal1")
    model.AddBoolAnd([pattern_sat_1, color_sat_1]).OnlyEnforceIf(both_sat_1)
    model.AddBoolOr([pattern_sat_1.Not(), color_sat_1.Not()]).OnlyEnforceIf(both_sat_1.Not())

    either_sat_1 = model.NewBoolVar("either_satisfied_goal1")
    model.AddBoolOr([pattern_sat_1, color_sat_1]).OnlyEnforceIf(either_sat_1)
    model.AddBoolAnd([pattern_sat_1.Not(), color_sat_1.Not()]).OnlyEnforceIf(either_sat_1.Not())

    partial_only_1 = model.NewBoolVar("partial_only_goal1")
    model.AddBoolAnd([either_sat_1, both_sat_1.Not()]).OnlyEnforceIf(partial_only_1)
    model.AddBoolOr([either_sat_1.Not(), both_sat_1]).OnlyEnforceIf(partial_only_1.Not())

    score_1 = model.NewIntVar(0, m2.higher_points, "score_goal1")
    model.Add(score_1 == m2.higher_points).OnlyEnforceIf(both_sat_1)
    model.Add(score_1 == m2.lower_points).OnlyEnforceIf(partial_only_1)
    model.Add(score_1 == 0).OnlyEnforceIf([both_sat_1.Not(), partial_only_1.Not()])

    design_goal_scores["goal_1"] = score_1

    # Goal 2: Similar logic
    pattern_sat_2 = add_soft_pattern_scoring(model, b_patterns_2, m3.config_numbers, "goal2_P")
    color_sat_2 = add_soft_pattern_scoring(model, b_colors_2, m3.config_numbers, "goal2_C")

    both_sat_2 = model.NewBoolVar("both_satisfied_goal2")
    model.AddBoolAnd([pattern_sat_2, color_sat_2]).OnlyEnforceIf(both_sat_2)
    model.AddBoolOr([pattern_sat_2.Not(), color_sat_2.Not()]).OnlyEnforceIf(both_sat_2.Not())

    either_sat_2 = model.NewBoolVar("either_satisfied_goal2")
    model.AddBoolOr([pattern_sat_2, color_sat_2]).OnlyEnforceIf(either_sat_2)
    model.AddBoolAnd([pattern_sat_2.Not(), color_sat_2.Not()]).OnlyEnforceIf(either_sat_2.Not())

    partial_only_2 = model.NewBoolVar("partial_only_goal2")
    model.AddBoolAnd([either_sat_2, both_sat_2.Not()]).OnlyEnforceIf(partial_only_2)
    model.AddBoolOr([either_sat_2.Not(), both_sat_2]).OnlyEnforceIf(partial_only_2.Not())

    score_2 = model.NewIntVar(0, m3.higher_points, "score_goal2")
    model.Add(score_2 == m3.higher_points).OnlyEnforceIf(both_sat_2)
    model.Add(score_2 == m3.lower_points).OnlyEnforceIf(partial_only_2)
    model.Add(score_2 == 0).OnlyEnforceIf([both_sat_2.Not(), partial_only_2.Not()])

    design_goal_scores["goal_2"] = score_2

    # Pair indicators for cap constraints (same as original)
    pair_indicators = add_pair_indicators(model, b_patterns, b_colors, "")

    for _k in range(k):
        for _l in range(k):
            model.Add(sum(pair_indicators[i][_k][_l] for i in range(len(pair_indicators))) <= cap)

    return ObjectiveDesignGoalsModel(
        model=model,
        pattern_variables=pattern_variables,
        color_variables=color_variables,
        z_pattern_variables=b_pattern_map,
        z_color_variables=b_color_map,
        pair_indicators=pair_indicators,
        variable_indices=variable_indices,
        k=k,
        design_goal_tiles=[m1, m2, m3],
        design_goal_scores=design_goal_scores,
        cap=cap,
        time_limit_s=time_limit_s,
    )


def main():
    """Example usage of the objective-based design goals solver."""
    v = [1, 2, 3, 4, 5, 6]

    # Create DesignGoalTile objects with the correct patterns from enum
    m1 = DesignGoalTiles.THREE_PAIRS.value  # [2, 2, 2] - AA-BB-CC
    m2 = DesignGoalTiles.THREE_TWO_ONE.value  # [3, 2, 1] - AAA-BB-C
    m3 = DesignGoalTiles.FOUR_TWO.value  # [4, 2] - AAAA-BB
    cap = 3

    print("=== Objective-based Design Goals Solver ===")
    print(f"Design Goal 1: {m1.config_name} (scores: {m1.lower_points}/{m1.higher_points})")
    print(f"Design Goal 2: {m2.config_name} (scores: {m2.lower_points}/{m2.higher_points})")
    print(f"Design Goal 3: {m3.config_name} (scores: {m3.lower_points}/{m3.higher_points})")
    print()

    solution = solve_objective_design_goals(v, m1, m2, m3, cap=cap, time_limit_s=10.0)
    if solution is None:
        print("No feasible solution found.")
    else:
        print("\n==== SUMMARY ====")
        print(f"Status: {solution['status']}")
        print(f"Total design goal score: {solution['total_score']}")
        print(f"Individual scores: {solution['design_goal_scores']}")


if __name__ == "__main__":
    main()
