# combined_patterns_paircap.py
# pip install ortools

from ortools.sat.python import cp_model

def add_channeling(model, G, values, tag):
    """Create channeling booleans b[i][k] <-> (G[i] == values[k]), with exactly-one per i."""
    n, K = len(G), len(values)
    b = [[model.NewBoolVar(f"b[{tag}][i={i},k={k}]") for k in range(K)] for i in range(n)]
    for i in range(n):
        model.Add(sum(b[i][k] for k in range(K)) == 1)
        for k in range(K):
            model.Add(G[i] == values[k]).OnlyEnforceIf(b[i][k])
            model.Add(G[i] != values[k]).OnlyEnforceIf(b[i][k].Not())
    return b

def add_pattern(model, b, mults, tag):
    """Enforce multiplicity pattern 'mults' over one set via slot assignment."""
    n, K = len(b), len(b[0])
    r = len(mults)
    w = [[model.NewBoolVar(f"w[{tag}][k={k},s={s}]") for s in range(r)] for k in range(K)]
    # each slot taken once
    for s in range(r):
        model.Add(sum(w[k][s] for k in range(K)) == 1)
    # each value serves at most one slot
    for k in range(K):
        model.Add(sum(w[k][s] for s in range(r)) <= 1)
    # count matching
    for k in range(K):
        occ_k = sum(b[i][k] for i in range(n))
        model.Add(occ_k == sum(mults[s] * w[k][s] for s in range(r)))
    # symmetry breaking among equal slots (optional speedup)
    eq = {}
    for s, sz in enumerate(mults):
        eq.setdefault(sz, []).append(s)
    for slots in eq.values():
        for a, bslot in zip(slots, slots[1:]):
            model.Add(sum((k+1) * w[k][a] for k in range(K))
                    <= sum((k+1) * w[k][bslot] for k in range(K)))
    return w

def add_pair_indicators(model, bU, bP, tag):
    """Create pair[i][k][l] = AND(bU[i][k], bP[i][l]); also exactly-one per i."""
    n, K = len(bU), len(bU[0])
    pair = [[[model.NewBoolVar(f"pair[{tag}][i={i},k={k},l={l}]")
              for l in range(K)] for k in range(K)] for i in range(n)]
    for i in range(n):
        model.Add(sum(pair[i][k][l] for k in range(K) for l in range(K)) == 1)
        for k in range(K):
            for l in range(K):
                # linear AND
                model.Add(pair[i][k][l] <= bU[i][k])
                model.Add(pair[i][k][l] <= bP[i][l])
                model.Add(pair[i][k][l] >= bU[i][k] + bP[i][l] - 1)
    return pair

def solve_combined(v, m1, m2, m3, cap=3, time_limit_s=5.0):
    """
    v  : list of 6 distinct ints
    m1 : pattern for X-sets (sum=6), e.g. [2,2,2] or [3,3] or [4,1,1]
    m2 : pattern for T-sets (sum=6)
    m3 : pattern for P-sets (sum=6)
    cap: max allowed count for any ordered pair across the 15 counted positions
    """
    assert len(v) == 6 and sum(m1) == 6 and sum(m2) == 6 and sum(m3) == 6
    K = 6
    n = 6

    model = cp_model.CpModel()
    dom = cp_model.Domain.FromValues(v)

    # Unprimed variables
    X  = [model.NewIntVarFromDomain(dom, f"X[{i}]") for i in range(n)]
    T  = [model.NewIntVarFromDomain(dom, f"T[{i}]") for i in range(n)]
    P  = [model.NewIntVarFromDomain(dom, f"P[{i}]") for i in range(n)]
    # Primed variables
    Xp = [model.NewIntVarFromDomain(dom, f"Xp[{i}]") for i in range(n)]
    Tp = [model.NewIntVarFromDomain(dom, f"Tp[{i}]") for i in range(n)]
    Pp = [model.NewIntVarFromDomain(dom, f"Pp[{i}]") for i in range(n)]

    # Channeling
    bX  = add_channeling(model, X,  v, "X")
    bT  = add_channeling(model, T,  v, "T")
    bP  = add_channeling(model, P,  v, "P")
    bXp = add_channeling(model, Xp, v, "Xp")
    bTp = add_channeling(model, Tp, v, "Tp")
    bPp = add_channeling(model, Pp, v, "Pp")

    # Patterns (same patterns on primed/unprimed sets)
    add_pattern(model, bX,  m1, "X")
    add_pattern(model, bT,  m2, "T")
    add_pattern(model, bP,  m3, "P")
    add_pattern(model, bXp, m1, "Xp")
    add_pattern(model, bTp, m2, "Tp")
    add_pattern(model, bPp, m3, "Pp")

    # Cross equalities (1-based in prompt → 0-based here)
    # Unprimed: x3=t1, x4=t6, x5=p2
    model.Add(X[2] == T[0])
    model.Add(X[3] == T[5])
    model.Add(X[4] == P[1])
    # Primed:   x'3=t'1, x'4=t'6, x'5=p'2
    model.Add(Xp[2] == Tp[0])
    model.Add(Xp[3] == Tp[5])
    model.Add(Xp[4] == Pp[1])

    # Pair indicators for (X[i],Xp[i]), (T[i],Tp[i]), (P[i],Pp[i])
    pairX = add_pair_indicators(model, bX,  bXp, "X")
    pairT = add_pair_indicators(model, bT,  bTp, "T")
    pairP = add_pair_indicators(model, bP,  bPp, "P")

    # Cap each ordered pair across the 15 counted positions
    IX = range(n)                      # all 6
    IT = [1,2,3,4]                     # exclude i=0 (t1) and i=5 (t6)
    IP = [0,2,3,4,5]                   # exclude i=1 (p2)

    for k in range(K):
        for l in range(K):
            model.Add(
                sum(pairX[i][k][l] for i in IX) +
                sum(pairT[i][k][l] for i in IT) +
                sum(pairP[i][k][l] for i in IP)
                <= cap
            )

    # Feasibility
    model.Minimize(0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Extract solutions
    sol = {
        "X":  [solver.Value(x) for x in X],
        "T":  [solver.Value(t) for t in T],
        "P":  [solver.Value(p) for p in P],
        "Xp": [solver.Value(x) for x in Xp],
        "Tp": [solver.Value(t) for t in Tp],
        "Pp": [solver.Value(p) for p in Pp],
    }

    # Optional: aggregate counted pair frequencies (for inspection)
    def count_pairs(pair, idxs):
        M = [[0]*K for _ in range(K)]
        for i in idxs:
            for k in range(K):
                for l in range(K):
                    M[k][l] += int(round(solver.Value(pair[i][k][l])))
        return M

    counts = [[0]*K for _ in range(K)]
    CX = count_pairs(pairX, IX)
    CT = count_pairs(pairT, IT)
    CP = count_pairs(pairP, IP)
    for k in range(K):
        for l in range(K):
            counts[k][l] = CX[k][l] + CT[k][l] + CP[k][l]

    sol["pair_counts_total"] = counts
    return sol

if __name__ == "__main__":
    # Example data
    v  = [3, 7, 11, 20, 25, 42]
    m1 = [2, 2, 2]   # X pattern: AA-BB-CC
    m2 = [3, 2, 1]   # T pattern: AAA-BB-C
    m3 = [4, 1, 1]   # P pattern: AAAA-B-C
    cap = 3

    ans = solve_combined(v, m1, m2, m3, cap=cap)
    if ans is None:
        print("Infeasible.")
    else:
        print("X :",  ans["X"])
        print("T :",  ans["T"])
        print("P :",  ans["P"])
        print("Xp:",  ans["Xp"])
        print("Tp:",  ans["Tp"])
        print("Pp:",  ans["Pp"])
        print("Pair counts across 15 positions (k rows, l cols):")
        for row in ans["pair_counts_total"]:
            print(row)
