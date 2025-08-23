# combined_patterns_paircap.py
# pip install ortools

from ortools.sat.python import cp_model
from pydantic import BaseModel

from core.enums.design_goal import DesignGoalTiles
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


def add_pattern(
    model: cp_model.CpModel, b: list[list[cp_model.IntVar]], mults: list[int], tag: str
) -> list[list[cp_model.IntVar]]:
    """Enforce multiplicity pattern 'mults' over one set via slot assignment."""
    n, k = len(b), len(b[0])
    r = len(mults)
    w = [[model.NewBoolVar(f"w[{tag}][k={k_val},s={s}]") for s in range(r)] for k_val in range(k)]
    # each slot taken once
    for s in range(r):
        model.Add(sum(w[k_val][s] for k_val in range(k)) == 1)
    # each value serves at most one slot
    for k_val in range(k):
        model.Add(sum(w[k_val][s] for s in range(r)) <= 1)
    # count matching
    for k_val in range(k):
        occ_k = sum(b[i][k_val] for i in range(n))
        model.Add(occ_k == sum(mults[s] * w[k_val][s] for s in range(r)))
    # symmetry breaking among equal slots (optional speedup)
    eq = {}
    for s, sz in enumerate(mults):
        eq.setdefault(sz, []).append(s)
    for slots in eq.values():
        for a, bslot in zip(slots, slots[1:], strict=False):
            model.Add(
                sum((k_val + 1) * w[k_val][a] for k_val in range(k))
                <= sum((k_val + 1) * w[k_val][bslot] for k_val in range(k))
            )
    return w


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


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Callback to print and collect all solutions."""

    def __init__(self, pattern_variables, color_variables, pair_indicators, variable_indices, k, design_goal_tiles):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._pattern_variables = pattern_variables  # dict containing pattern variables
        self._color_variables = color_variables  # dict containing color variables
        self._pair_indicators = pair_indicators  # pair indicators array
        self._variable_indices = variable_indices  # list of variable indices
        self._k = k
        self._design_goal_tiles = design_goal_tiles  # the three design goal tiles
        self._solutions = []
        self._solution_count = 0

    def _create_quilt_board_from_solution(self, pattern_values, color_values):
        """Create a quilt board from the solution's pattern and color values."""
        from core.enums.color import ALL_COLORS
        from core.enums.pattern import ALL_PATTERNS
        from core.models.patch_tile import PatchTile
        from core.models.quilt_board import HexPosition, QuiltBoard

        # Create mapping from numeric values [1,2,3,4,5,6] to actual enums
        # Assuming v = [1,2,3,4,5,6] maps to the first 6 colors and patterns
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


class DesignGoalsModel(BaseModel):
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
    cap: int
    time_limit_s: float


def solve_combined(
    v: list[int], m1: DesignGoalTile, m2: DesignGoalTile, m3: DesignGoalTile, cap: int = 3, time_limit_s: float = 5.0
):
    design_goals_model = build_model(None, v, m1, m2, m3, cap, time_limit_s)

    # Create solution callback
    solution_printer = SolutionPrinter(
        design_goals_model.pattern_variables,
        design_goals_model.color_variables,
        design_goals_model.pair_indicators,
        design_goals_model.variable_indices,
        design_goals_model.k,
        design_goals_model.design_goal_tiles,
    )

    # Search for all solutions
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.enumerate_all_solutions = True

    print("Searching for all feasible solutions...")
    status = solver.SearchForAllSolutions(design_goals_model.model, solution_printer)

    print(f"\nSearch completed with status: {solver.StatusName(status)}")
    print(f"Total solutions found: {solution_printer.get_solution_count()}")

    if solution_printer.get_solution_count() == 0:
        return None
    return solution_printer.get_solutions()


def build_model(
    model: cp_model.CpModel | None = None,
    v: list[int] = None,
    m1: DesignGoalTile = DesignGoalTiles.THREE_PAIRS.value,
    m2: DesignGoalTile = DesignGoalTiles.THREE_TWO_ONE.value,
    m3: DesignGoalTile = DesignGoalTiles.FOUR_TWO.value,
    cap: int = 3,
    time_limit_s: float = 5.0,
) -> DesignGoalsModel:
    """
    v  : list of 6 distinct ints
    m1 : pattern for X-sets (sum=6), e.g. [2,2,2] or [3,3] or [4,1,1]
    m2 : pattern for T-sets (sum=6)
    m3 : pattern for P-sets (sum=6)
    cap: max allowed count for any ordered pair across the 15 counted positions

    Returns all feasible solutions.
    """
    if v is None:
        v = [1, 2, 3, 4, 5, 6]
    assert len(v) == 6
    k = 6

    if not model:
        model = cp_model.CpModel()

    dom = cp_model.Domain.FromValues(v)

    board = QuiltBoard(design_goal_tiles=[m1, m2, m3])
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

    # goal tile 0
    design_goal_patch_tiles_0 = board.get_design_goal_patch_tiles(0)
    variable_indices_0 = [i.abs for i in design_goal_patch_tiles_0]
    b_patterns_0 = [b_pattern_map[i] for i in variable_indices_0]
    b_colors_0 = [b_color_map[i] for i in variable_indices_0]

    # goal tile 1
    design_goal_patch_tiles_1 = board.get_design_goal_patch_tiles(1)
    variable_indices_1 = [i.abs for i in design_goal_patch_tiles_1]
    b_patterns_1 = [b_pattern_map[i] for i in variable_indices_1]
    b_colors_1 = [b_color_map[i] for i in variable_indices_1]

    # goal tile 2
    design_goal_patch_tiles_2 = board.get_design_goal_patch_tiles(2)
    variable_indices_2 = [i.abs for i in design_goal_patch_tiles_2]
    b_patterns_2 = [b_pattern_map[i] for i in variable_indices_2]
    b_colors_2 = [b_color_map[i] for i in variable_indices_2]

    # bx = add_channeling(model, x, v, "X")
    # bt = add_channeling(model, t, v, "T")
    # bp = add_channeling(model, p, v, "P")
    # bxp = add_channeling(model, xp, v, "Xp")
    # btp = add_channeling(model, tp, v, "Tp")
    # bpp = add_channeling(model, pp, v, "Pp")

    # Patterns (same patterns on primed/unprimed sets)
    add_pattern(model, b_patterns_0, m1.config_numbers, "P")
    add_pattern(model, b_patterns_1, m2.config_numbers, "P")
    add_pattern(model, b_patterns_2, m3.config_numbers, "P")
    add_pattern(model, b_colors_0, m1.config_numbers, "C")
    add_pattern(model, b_colors_1, m2.config_numbers, "C")
    add_pattern(model, b_colors_2, m3.config_numbers, "C")

    # # Cross equalities (1-based in prompt → 0-based here)
    # # Unprimed: x3=t1, x4=t6, x5=p2
    # model.Add(x[2] == t[0])
    # model.Add(x[3] == t[5])
    # model.Add(x[4] == p[1])
    # # Primed:   x'3=t'1, x'4=t'6, x'5=p'2
    # model.Add(xp[2] == tp[0])
    # model.Add(xp[3] == tp[5])
    # model.Add(xp[4] == pp[1])

    pair_indicators = add_pair_indicators(model, b_patterns, b_colors, "")

    # # Pair indicators for (x[i],xp[i]), (t[i],tp[i]), (p[i],pp[i])
    # pair_x = add_pair_indicators(model, bx, bxp, "X")
    # pair_t = add_pair_indicators(model, bt, btp, "T")
    # pair_p = add_pair_indicators(model, bp, bpp, "P")

    # # Cap each ordered pair across the 15 counted positions
    # ix = range(n)  # all 6
    # it = [1, 2, 3, 4]  # exclude i=0 (t1) and i=5 (t6)
    # ip = [0, 2, 3, 4, 5]  # exclude i=1 (p2)

    for _k in range(k):
        for _l in range(k):
            model.Add(sum(pair_indicators[i][_k][_l] for i in range(len(pair_indicators))) <= cap)

    return DesignGoalsModel(
        model=model,
        pattern_variables=pattern_variables,
        color_variables=color_variables,
        z_pattern_variables=b_pattern_map,
        z_color_variables=b_color_map,
        pair_indicators=pair_indicators,
        variable_indices=variable_indices,
        k=k,
        design_goal_tiles=[m1, m2, m3],
        cap=cap,
        time_limit_s=time_limit_s,
    )


def main():
    v = [1, 2, 3, 4, 5, 6]

    # Create DesignGoalTile objects with the correct patterns from enum

    m1 = DesignGoalTiles.THREE_PAIRS.value  # [2, 2, 2] - AA-BB-CC
    m2 = DesignGoalTiles.THREE_TWO_ONE.value  # [3, 2, 1] - AAA-BB-C
    m3 = DesignGoalTiles.FOUR_TWO.value  # [4, 2] - AAAA-BB (closest match)
    cap = 3

    solutions = solve_combined(v, m1, m2, m3, cap=cap)
    if solutions is None:
        print("No feasible solutions found.")
    else:
        print("\n==== SUMMARY ====")
        print(f"Found {len(solutions)} total feasible solutions.")


if __name__ == "__main__":
    # Example data
    main()
