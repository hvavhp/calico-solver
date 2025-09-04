# solve_three_patterns.py
# Requires: pip install ortools

from ortools.sat.python import cp_model


def solve_three_patterns(v, m1, m2, m3):
    """
    v  : list of 6 distinct integers (the allowed values), e.g. [3,7,11,20,25,42]
    m1 : multiplicities for (x_1..x_6),   e.g. [2,2,2] or [3,3] or [4,1,1] etc., sum=6
    m2 : multiplicities for (t_1..t_6),   sum=6
    m3 : multiplicities for (p_1..p_6),   sum=6
    Returns one feasible assignment (x, t, p) or None if infeasible.
    """
    no_values = len(v)
    n = 6
    assert no_values == 6 and n == 6, "This helper expects K=n=6"
    assert sum(m1) == n and sum(m2) == n and sum(m3) == n

    model = cp_model.CpModel()
    dom = cp_model.Domain.FromValues(v)

    # Variables (domain over v)
    x = [model.NewIntVarFromDomain(dom, f"x[{i}]") for i in range(n)]
    t = [model.NewIntVarFromDomain(dom, f"t[{i}]") for i in range(n)]
    p = [model.NewIntVarFromDomain(dom, f"p[{i}]") for i in range(n)]

    # Channeling booleans: bG[i][k] == 1  <=>  G[i] == v[k]
    def make_b(group, name):
        return [[model.NewBoolVar(f"{name}_is_v{v[k]}[{i}]") for k in range(no_values)] for i in range(n)]

    b_x = make_b(x, "x")
    b_t = make_b(t, "t")
    b_p = make_b(p, "p")

    # For each group, exactly one value chosen per position, and channel equality
    def attach_channel(group, b_g):
        for i in range(n):
            model.Add(sum(b_g[i][k] for k in range(no_values)) == 1)
            for k in range(no_values):
                model.Add(group[i] == v[k]).OnlyEnforceIf(b_g[i][k])
                model.Add(group[i] != v[k]).OnlyEnforceIf(b_g[i][k].Not())

    attach_channel(x, b_x)
    attach_channel(t, b_t)
    attach_channel(p, b_p)

    # Pattern slot assignment: for group G with multiplicities m
    # wG[k][slot] = 1 if value v[k] is used for slot "slot" (size m[slot])
    def add_pattern_constraints(b_g, m, tag):
        r = len(m)
        w = [[model.NewBoolVar(f"w_{tag}[k={k},slot={s}]") for s in range(r)] for k in range(no_values)]

        # each slot is taken by exactly one value
        for s in range(r):
            model.Add(sum(w[k][s] for k in range(no_values)) == 1)

        # each value serves at most one slot
        for k in range(no_values):
            model.Add(sum(w[k][s] for s in range(r)) <= 1)

        # count-matching: occurrences of v[k] equals the size of the slot it takes (or 0)
        for k in range(no_values):
            occ_k = sum(b_g[i][k] for i in range(n))
            model.Add(occ_k == sum(m[s] * w[k][s] for s in range(r)))

        # optional symmetry breaking for equal multiplicities (speedup; not required)
        # enforce nondecreasing chosen k for equal-sized slots
        eq = {}
        for s, size in enumerate(m):
            eq.setdefault(size, []).append(s)
        for _, slots in eq.items():
            for a, b in zip(slots, slots[1:], strict=False):
                model.Add(
                    sum((k + 1) * w[k][a] for k in range(no_values)) <= sum((k + 1) * w[k][b] for k in range(no_values))
                )
        return w

    add_pattern_constraints(b_x, m1, "X")
    add_pattern_constraints(b_t, m2, "T")
    add_pattern_constraints(b_p, m3, "P")

    # Cross-group links (1-based indices in your spec → 0-based here)
    model.Add(x[2] == t[0])  # x3 = t1
    model.Add(x[3] == t[5])  # x4 = t6
    model.Add(x[4] == p[1])  # x5 = p2

    # Feasibility problem
    model.Minimize(0)
    solver = cp_model.CpSolver()
    # This is tiny; default params are fine. You can set a small limit if you want:
    # solver.parameters.max_time_in_seconds = 5.0

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    return (
        [solver.Value(xi) for xi in x],
        [solver.Value(ti) for ti in t],
        [solver.Value(pi) for pi in p],
    )


class _AllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    """Collects all (x, t, p) assignments found by the CP-SAT solver.

    Use with `CpSolver().SearchForAllSolutions(model, collector)`.
    """

    def __init__(self, x_vars, t_vars, p_vars, limit: int | None = None):
        super().__init__()
        self._x_vars = list(x_vars)
        self._t_vars = list(t_vars)
        self._p_vars = list(p_vars)
        self._solutions: list[tuple[list[int], list[int], list[int]]] = []
        self._limit = limit

    def on_solution_callback(self) -> None:  # noqa: N802 (OR-Tools naming)
        x_sol = [self.Value(v) for v in self._x_vars]
        t_sol = [self.Value(v) for v in self._t_vars]
        p_sol = [self.Value(v) for v in self._p_vars]
        self._solutions.append((x_sol, t_sol, p_sol))
        if self._limit is not None and len(self._solutions) >= self._limit:
            self.StopSearch()

    @property
    def solutions(self) -> list[tuple[list[int], list[int], list[int]]]:
        return self._solutions


def solve_three_patterns_all(v, m1, m2, m3, max_solutions: int | None = None):
    """Enumerate all feasible assignments (x, t, p) or up to `max_solutions`.

    Returns a list of solutions; each solution is a tuple of three lists: (x, t, p).

    Warning: The number of solutions can be very large. Use `max_solutions`
    to cap the enumeration if needed.
    """
    no_values = len(v)
    n = 6
    assert no_values == 6 and n == 6, "This helper expects K=n=6"
    assert sum(m1) == n and sum(m2) == n and sum(m3) == n

    model = cp_model.CpModel()
    dom = cp_model.Domain.FromValues(v)

    x = [model.NewIntVarFromDomain(dom, f"x[{i}]") for i in range(n)]
    t = [model.NewIntVarFromDomain(dom, f"t[{i}]") for i in range(n)]
    p = [model.NewIntVarFromDomain(dom, f"p[{i}]") for i in range(n)]

    def make_b(group, name):
        return [[model.NewBoolVar(f"{name}_is_v{v[k]}[{i}]") for k in range(no_values)] for i in range(n)]

    b_x = make_b(x, "x")
    b_t = make_b(t, "t")
    b_p = make_b(p, "p")

    def attach_channel(group, b_g):
        for i in range(n):
            model.Add(sum(b_g[i][k] for k in range(no_values)) == 1)
            for k in range(no_values):
                model.Add(group[i] == v[k]).OnlyEnforceIf(b_g[i][k])
                model.Add(group[i] != v[k]).OnlyEnforceIf(b_g[i][k].Not())

    attach_channel(x, b_x)
    attach_channel(t, b_t)
    attach_channel(p, b_p)

    def add_pattern_constraints(b_g, m, tag):
        r = len(m)
        w = [[model.NewBoolVar(f"w_{tag}[k={k},slot={s}]") for s in range(r)] for k in range(no_values)]
        for s in range(r):
            model.Add(sum(w[k][s] for k in range(no_values)) == 1)
        for k in range(no_values):
            model.Add(sum(w[k][s] for s in range(r)) <= 1)
        for k in range(no_values):
            occ_k = sum(b_g[i][k] for i in range(n))
            model.Add(occ_k == sum(m[s] * w[k][s] for s in range(r)))
        eq: dict[int, list[int]] = {}
        for s, size in enumerate(m):
            eq.setdefault(size, []).append(s)
        for _, slots in eq.items():
            for a, b in zip(slots, slots[1:], strict=False):
                model.Add(
                    sum((k + 1) * w[k][a] for k in range(no_values)) <= sum((k + 1) * w[k][b] for k in range(no_values))
                )
        return w

    add_pattern_constraints(b_x, m1, "X")
    add_pattern_constraints(b_t, m2, "T")
    add_pattern_constraints(b_p, m3, "P")

    model.Add(x[2] == t[0])
    model.Add(x[3] == t[5])
    model.Add(x[4] == p[1])

    # No objective needed for satisfiability enumeration
    solver = cp_model.CpSolver()
    collector = _AllSolutionsCollector(x, t, p, limit=max_solutions)
    solver.SearchForAllSolutions(model, collector)

    # If no solutions were found, return an empty list to signal infeasibility.
    if not collector.solutions:
        return []
    return collector.solutions


def _canonical_form(
    solution: tuple[list[int], list[int], list[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Convert solution to canonical form by mapping values to standardized labels.

    Two solutions are equivalent if they have the same canonical form.
    The canonical form uses consecutive integers 0, 1, 2, ... based on
    the order of first appearance of values in the combined (x, t, p) sequence.
    """
    x_sol, t_sol, p_sol = solution

    # Combine all values in order of appearance
    all_values = x_sol + t_sol + p_sol

    # Create mapping from original values to canonical labels (0, 1, 2, ...)
    value_to_label = {}
    next_label = 0

    for val in all_values:
        if val not in value_to_label:
            value_to_label[val] = next_label
            next_label += 1

    # Convert solution to canonical form
    canonical_x = tuple(value_to_label[val] for val in x_sol)
    canonical_t = tuple(value_to_label[val] for val in t_sol)
    canonical_p = tuple(value_to_label[val] for val in p_sol)

    return (canonical_x, canonical_t, canonical_p)


def deduplicate_solutions(
    solutions: list[tuple[list[int], list[int], list[int]]],
) -> list[tuple[list[int], list[int], list[int]]]:
    """Remove solutions that are equivalent under value substitution.

    Two solutions are considered equivalent if one can be obtained from the other
    by consistently swapping values while preserving the pattern structure.
    """
    if not solutions:
        return solutions

    seen_canonical = set()
    unique_solutions = []

    for solution in solutions:
        canonical = _canonical_form(solution)
        if canonical not in seen_canonical:
            seen_canonical.add(canonical)
            unique_solutions.append(solution)

    return unique_solutions


def solve_three_patterns_all_unique(v, m1, m2, m3, max_solutions: int | None = None):
    """Enumerate all unique feasible assignments (x, t, p) up to value substitution.

    Returns deduplicated solutions where equivalent patterns are removed.
    Two solutions are equivalent if one can be obtained from the other by
    consistently swapping values.
    """
    # Get all solutions first
    all_solutions = solve_three_patterns_all(v, m1, m2, m3, max_solutions)

    # Deduplicate based on canonical forms
    return deduplicate_solutions(all_solutions)


if __name__ == "__main__":
    # Example usage
    v = [3, 7, 11, 20, 25, 42]  # the 6 distinct integers (K=6)
    m1 = [1, 1, 1, 1, 1, 1]
    m2 = [3, 2, 1]
    m3 = [4, 1, 1]

    print("One solution (if any):")
    one = solve_three_patterns(v, m1, m2, m3)
    if one is None:
        print("No feasible assignment.")
    else:
        x_sol, t_sol, p_sol = one
        print("x:", x_sol)
        print("t:", t_sol)
        print("p:", p_sol)
        print("Check links:", x_sol[2] == t_sol[0], x_sol[3] == t_sol[5], x_sol[4] == p_sol[1])

    print("\nEnumerating all solutions (capped at 50):")
    all_solutions = solve_three_patterns_all(v, m1, m2, m3, max_solutions=50)
    if not all_solutions:
        print("No feasible assignment.")
    else:
        print(f"Found {len(all_solutions)} total solutions")
        for idx, (x_sol, t_sol, p_sol) in enumerate(all_solutions[:5], start=1):  # Show first 5
            print(f"Solution {idx}:")
            print("  x:", x_sol)
            print("  t:", t_sol)
            print("  p:", p_sol)
        if len(all_solutions) > 5:
            print(f"... and {len(all_solutions) - 5} more solutions")

    print("\nEnumerating unique solutions (removing equivalent patterns):")
    unique_solutions = solve_three_patterns_all_unique(v, m1, m2, m3, max_solutions=50000)
    if not unique_solutions:
        print("No feasible assignment.")
    else:
        print(f"Found {len(unique_solutions)} unique solutions (out of {len(all_solutions)} total)")
        for idx, (x_sol, t_sol, p_sol) in enumerate(unique_solutions, start=1):
            print(f"Unique solution {idx}:")
            print("  x:", x_sol)
            print("  t:", t_sol)
            print("  p:", p_sol)
