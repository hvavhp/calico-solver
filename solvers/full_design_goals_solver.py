# combined_patterns_paircap.py
# pip install ortools

from ortools.sat.python import cp_model


def add_channeling(model, g, values, tag):
    """Create channeling booleans b[i][k] <-> (g[i] == values[k]), with exactly-one per i."""
    n, k = len(g), len(values)
    b = [[model.NewBoolVar(f"b[{tag}][i={i},k={k_val}]") for k_val in range(k)] for i in range(n)]
    for i in range(n):
        model.Add(sum(b[i][k_val] for k_val in range(k)) == 1)
        for k_val in range(k):
            model.Add(g[i] == values[k_val]).OnlyEnforceIf(b[i][k_val])
            model.Add(g[i] != values[k_val]).OnlyEnforceIf(b[i][k_val].Not())
    return b


def add_pattern(model, b, mults, tag):
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


def add_pair_indicators(model, bu, bp, tag):
    """Create pair[i][k][val] = AND(bu[i][k], bp[i][val]); also exactly-one per i."""
    n, k = len(bu), len(bu[0])
    pair = [
        [[model.NewBoolVar(f"pair[{tag}][i={i},k={k_val},val={val}]") for val in range(k)] for k_val in range(k)]
        for i in range(n)
    ]
    for i in range(n):
        model.Add(sum(pair[i][k_val][val] for k_val in range(k) for val in range(k)) == 1)
        for k_val in range(k):
            for val in range(k):
                # linear AND
                model.Add(pair[i][k_val][val] <= bu[i][k_val])
                model.Add(pair[i][k_val][val] <= bp[i][val])
                model.Add(pair[i][k_val][val] >= bu[i][k_val] + bp[i][val] - 1)
    return pair


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Callback to print and collect all solutions."""

    def __init__(self, variables, pair_variables, k, ix, it, ip):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._variables = variables  # dict containing x, t, p, xp, tp, pp
        self._pair_variables = pair_variables  # dict containing pair_x, pair_t, pair_p
        self._k = k
        self._ix = ix
        self._it = it
        self._ip = ip
        self._solutions = []
        self._solution_count = 0

    def on_solution_callback(self):
        self._solution_count += 1
        print(f"\n--- Solution {self._solution_count} ---")

        # Extract variable values
        sol = {}
        for name, vars_list in self._variables.items():
            sol[name] = [self.Value(var) for var in vars_list]
            print(f"{name}: {sol[name]}")

        # Extract pair counts
        def count_pairs(pair, idxs):
            m = [[0] * self._k for _ in range(self._k)]
            for i in idxs:
                for k_val in range(self._k):
                    for val in range(self._k):
                        m[k_val][val] += int(round(self.Value(pair[i][k_val][val])))
            return m

        counts = [[0] * self._k for _ in range(self._k)]
        cx = count_pairs(self._pair_variables["pair_x"], self._ix)
        ct = count_pairs(self._pair_variables["pair_t"], self._it)
        cp = count_pairs(self._pair_variables["pair_p"], self._ip)
        for k_val in range(self._k):
            for val in range(self._k):
                counts[k_val][val] = cx[k_val][val] + ct[k_val][val] + cp[k_val][val]

        sol["pair_counts_total"] = counts
        print("Pair counts across 15 positions (k rows, l cols):")
        for row in counts:
            print(row)

        self._solutions.append(sol)

    def get_solutions(self):
        return self._solutions

    def get_solution_count(self):
        return self._solution_count


def solve_combined(v, m1, m2, m3, cap=3, time_limit_s=5.0):
    """
    v  : list of 6 distinct ints
    m1 : pattern for X-sets (sum=6), e.g. [2,2,2] or [3,3] or [4,1,1]
    m2 : pattern for T-sets (sum=6)
    m3 : pattern for P-sets (sum=6)
    cap: max allowed count for any ordered pair across the 15 counted positions

    Returns all feasible solutions.
    """
    assert len(v) == 6 and sum(m1) == 6 and sum(m2) == 6 and sum(m3) == 6
    k = 6
    n = 6

    model = cp_model.CpModel()
    dom = cp_model.Domain.FromValues(v)

    # Unprimed variables
    x = [model.NewIntVarFromDomain(dom, f"X[{i}]") for i in range(n)]
    t = [model.NewIntVarFromDomain(dom, f"T[{i}]") for i in range(n)]
    p = [model.NewIntVarFromDomain(dom, f"P[{i}]") for i in range(n)]
    # Primed variables
    xp = [model.NewIntVarFromDomain(dom, f"Xp[{i}]") for i in range(n)]
    tp = [model.NewIntVarFromDomain(dom, f"Tp[{i}]") for i in range(n)]
    pp = [model.NewIntVarFromDomain(dom, f"Pp[{i}]") for i in range(n)]

    # Channeling
    bx = add_channeling(model, x, v, "X")
    bt = add_channeling(model, t, v, "T")
    bp = add_channeling(model, p, v, "P")
    bxp = add_channeling(model, xp, v, "Xp")
    btp = add_channeling(model, tp, v, "Tp")
    bpp = add_channeling(model, pp, v, "Pp")

    # Patterns (same patterns on primed/unprimed sets)
    add_pattern(model, bx, m1, "X")
    add_pattern(model, bt, m2, "T")
    add_pattern(model, bp, m3, "P")
    add_pattern(model, bxp, m1, "Xp")
    add_pattern(model, btp, m2, "Tp")
    add_pattern(model, bpp, m3, "Pp")

    # Cross equalities (1-based in prompt → 0-based here)
    # Unprimed: x3=t1, x4=t6, x5=p2
    model.Add(x[2] == t[0])
    model.Add(x[3] == t[5])
    model.Add(x[4] == p[1])
    # Primed:   x'3=t'1, x'4=t'6, x'5=p'2
    model.Add(xp[2] == tp[0])
    model.Add(xp[3] == tp[5])
    model.Add(xp[4] == pp[1])

    # Pair indicators for (x[i],xp[i]), (t[i],tp[i]), (p[i],pp[i])
    pair_x = add_pair_indicators(model, bx, bxp, "X")
    pair_t = add_pair_indicators(model, bt, btp, "T")
    pair_p = add_pair_indicators(model, bp, bpp, "P")

    # Cap each ordered pair across the 15 counted positions
    ix = range(n)  # all 6
    it = [1, 2, 3, 4]  # exclude i=0 (t1) and i=5 (t6)
    ip = [0, 2, 3, 4, 5]  # exclude i=1 (p2)

    for k_val in range(k):
        for val in range(k):
            model.Add(
                sum(pair_x[i][k_val][val] for i in ix)
                + sum(pair_t[i][k_val][val] for i in it)
                + sum(pair_p[i][k_val][val] for i in ip)
                <= cap
            )

    # Create solution callback
    variables = {"X": x, "T": t, "P": p, "Xp": xp, "Tp": tp, "Pp": pp}
    pair_variables = {"pair_x": pair_x, "pair_t": pair_t, "pair_p": pair_p}
    solution_printer = SolutionPrinter(variables, pair_variables, k, ix, it, ip)

    # Search for all solutions
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.enumerate_all_solutions = True

    print("Searching for all feasible solutions...")
    status = solver.SearchForAllSolutions(model, solution_printer)

    print(f"\nSearch completed with status: {solver.StatusName(status)}")
    print(f"Total solutions found: {solution_printer.get_solution_count()}")

    if solution_printer.get_solution_count() == 0:
        return None
    return solution_printer.get_solutions()


if __name__ == "__main__":
    # Example data
    v = [1, 2, 3, 4, 5, 6]
    m1 = [2, 2, 2]  # X pattern: AA-BB-CC
    m2 = [3, 2, 1]  # T pattern: AAA-BB-C
    m3 = [4, 1, 1]  # P pattern: AAAA-B-C
    cap = 3

    solutions = solve_combined(v, m1, m2, m3, cap=cap)
    if solutions is None:
        print("No feasible solutions found.")
    else:
        print("\n==== SUMMARY ====")
        print(f"Found {len(solutions)} total feasible solutions.")
