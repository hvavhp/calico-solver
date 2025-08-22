from ortools.sat.python import cp_model


def build_model(n, values, m, edges, add_additional_constraints=None):
    """
    n: number of base vars x[0..n-1]
    values: [v1,..,vK] (K can be 6 as in your case)
    m: list of subset specs over indices of x; each item either:
       - {"first":[...], "second":[...]}  OR
       - {"subset":[...], "t": t} meaning first=subset[:t], second=subset[t:].
    edges: list of undirected edges on subset indices, e.g. [(0,1),(1,2),...]
    add_additional_constraints: optional callback(model, x, values) to add your original constraints on x.
    """
    model = cp_model.CpModel()
    k = len(values)
    s = len(m)

    # 1) Decision vars: x_i in allowed set
    x = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(values), f"x[{i}]") for i in range(n)]

    # 2) Literals eq[(i,k)] <=> (x[i] == values[k])
    eq = {}
    for i in range(n):
        for _k, val in enumerate(values):
            b = model.NewBoolVar(f"eq_x[{i}]_v{_k}")
            model.Add(x[i] == val).OnlyEnforceIf(b)
            model.Add(x[i] != val).OnlyEnforceIf(b.Not())
            eq[(i, _k)] = b

    # 3) Parse subset descriptions
    split_sets = []
    for item in m:
        if "first" in item and "second" in item:
            first, second = item["first"], item["second"]
        else:
            subset, t = item["subset"], item["t"]
            assert 1 <= t <= len(subset), "Require t>=1"
            first, second = subset[:t], subset[t:]
        split_sets.append((first, second))

    # 4) Compute y_{S,k} and y_S
    y_sk = [[None for _ in range(k)] for _ in range(s)]
    y_s = [model.NewBoolVar(f"y_subset[{_s}]") for _s in range(s)]

    for _s, (first, second) in enumerate(split_sets):
        # One literal per k saying subset s wins with value v_k
        sum_k = []
        for _k in range(k):
            conds = [eq[(i, _k)] for i in first] + [eq[(j, _k)].Not() for j in second]
            y_sk[_s][_k] = model.NewBoolVar(f"y_subset[{_s}]_val{_k}")
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
    r = [[model.NewBoolVar(f"r[{_r}][{_k}]") for _k in range(k)] for _r in range(s)]
    # a[S][R][k] == 1  <=> S belongs to representative R in label k
    a = [[[model.NewBoolVar(f"a[{_s}][{_r}][{_k}]") for _k in range(k)] for _r in range(s)] for _s in range(s)]

    # Every winning vertex (for k) picks exactly one representative (for k)
    for _s in range(s):
        for _k in range(k):
            model.Add(sum(a[_s][_r][_k] for _r in range(s)) == y_sk[_s][_k])

    # Representative consistency and self-labeling
    for _r in range(s):
        for _k in range(k):
            model.Add(a[_r][_r][_k] == r[_r][_k])
            for _s in range(s):
                model.Add(a[_s][_r][_k] <= r[_r][_k])

    # Edge-induced equality (only when both endpoints win with the same k)
    for u, v in edges:
        for _r in range(s):
            for _k in range(k):
                model.Add(a[u][_r][_k] == a[v][_r][_k]).OnlyEnforceIf([y_sk[u][_k], y_sk[v][_k]])

    # Objective: total number of k-consistent activated components
    model.Maximize(sum(r[_r][_k] for _r in range(s) for _k in range(k)))

    # 6) Your original constraints on x
    if add_additional_constraints:
        add_additional_constraints(model, x, values)

    return model, x, y_s, y_sk, r, a


def solve_model(model, x, y_s, y_sk, r, a, time_limit_sec=None, workers=8):
    solver = cp_model.CpSolver()
    if time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = workers

    status = solver.Solve(model)
    out = {"status": solver.StatusName(status)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["objective"] = solver.ObjectiveValue()
        out["x"] = [solver.Value(v) for v in x]
        out["yS"] = [solver.Value(v) for v in y_s]
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
    return out


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Base variables and allowed values
    n = 25
    values = [10, 20, 30, 40, 50, 60]  # your v1..v6

    # Generate 100 subsets with larger sizes and larger first groups
    m = [
        # First 20 subsets (original pattern)
        {"subset": [0, 1, 2, 3, 4, 5], "t": 4},  # first=[0,1,2,3], second=[4,5]
        {"first": [2, 5, 8, 11], "second": [6, 7, 1]},  # explicit with larger first group
        {"subset": [4, 6, 0, 9, 12], "t": 3},  # first=[4,6,0], second=[9,12]
        {"subset": [8, 9, 10, 11, 13, 14], "t": 4},  # first=[8,9,10,11], second=[13,14]
        {"subset": [12, 13, 14, 15, 16], "t": 3},  # first=[12,13,14], second=[15,16]
        {"subset": [15, 16, 17, 18, 19, 20], "t": 4},  # first=[15,16,17,18], second=[19,20]
        {"subset": [19, 20, 21, 22, 23], "t": 3},  # first=[19,20,21], second=[22,23]
        {"subset": [22, 23, 24, 3, 7, 11], "t": 4},  # first=[22,23,24,3], second=[7,11]
        {"subset": [3, 7, 11, 17, 21], "t": 3},  # first=[3,7,11], second=[17,21]
        {"subset": [5, 9, 13, 17, 1, 18], "t": 4},  # first=[5,9,13,17], second=[1,18]
        {"subset": [1, 4, 8, 12, 16, 20], "t": 4},  # first=[1,4,8,12], second=[16,20]
        {"subset": [14, 18, 22, 6, 10], "t": 3},  # first=[14,18,22], second=[6,10]
        {"subset": [16, 20, 24, 0, 4, 8], "t": 4},  # first=[16,20,24,0], second=[4,8]
        {"subset": [2, 6, 10, 14, 19, 23], "t": 4},  # first=[2,6,10,14], second=[19,23]
        {"subset": [15, 19, 23, 1, 5], "t": 3},  # first=[15,19,23], second=[1,5]
        {"subset": [0, 8, 16, 24, 2, 10], "t": 4},  # first=[0,8,16,24], second=[2,10]
        {"subset": [1, 9, 17, 21, 13, 7], "t": 4},  # first=[1,9,17,21], second=[13,7]
        {"subset": [3, 11, 19, 4, 12], "t": 3},  # first=[3,11,19], second=[4,12]
        {"subset": [5, 13, 21, 18, 22, 6], "t": 4},  # first=[5,13,21,18], second=[22,6]
        {"subset": [7, 15, 23, 9, 24], "t": 3},  # first=[7,15,23], second=[9,24]
        # Additional 80 subsets to reach 100 total
        {"subset": [0, 3, 6, 9, 12, 15], "t": 4},
        {"subset": [1, 4, 7, 10, 13, 16], "t": 3},
        {"subset": [2, 5, 8, 11, 14, 17], "t": 4},
        {"subset": [18, 19, 20, 21, 22], "t": 3},
        {"subset": [23, 24, 0, 1, 2], "t": 3},
        {"first": [3, 4, 5, 6], "second": [7, 8, 9]},
        {"subset": [10, 11, 12, 13, 14, 15], "t": 4},
        {"subset": [16, 17, 18, 19, 20, 21], "t": 3},
        {"subset": [22, 23, 24, 0, 1, 2], "t": 4},
        {"subset": [3, 6, 9, 12, 15, 18], "t": 3},
        {"subset": [21, 24, 2, 5, 8, 11], "t": 4},
        {"subset": [14, 17, 20, 23, 1, 4], "t": 3},
        {"first": [7, 10, 13, 16], "second": [19, 22, 0]},
        {"subset": [3, 7, 11, 15, 19, 23], "t": 4},
        {"subset": [2, 6, 10, 14, 18, 22], "t": 3},
        {"subset": [1, 5, 9, 13, 17, 21], "t": 4},
        {"subset": [0, 4, 8, 12, 16, 20], "t": 3},
        {"subset": [24, 3, 7, 11, 15], "t": 3},
        {"subset": [19, 23, 2, 6, 10], "t": 4},
        {"subset": [14, 18, 22, 1, 5], "t": 3},
        {"subset": [9, 13, 17, 21, 0, 4], "t": 4},
        {"first": [8, 12, 16, 20], "second": [24, 3]},
        {"subset": [7, 11, 15, 19, 23, 2], "t": 3},
        {"subset": [6, 10, 14, 18, 22, 1], "t": 4},
        {"subset": [5, 9, 13, 17, 21, 0], "t": 3},
        {"subset": [4, 8, 12, 16, 20, 24], "t": 4},
        {"subset": [3, 7, 11, 15, 19], "t": 3},
        {"subset": [23, 2, 6, 10, 14], "t": 4},
        {"subset": [18, 22, 1, 5, 9], "t": 3},
        {"subset": [13, 17, 21, 0, 4, 8], "t": 4},
        {"first": [12, 16, 20, 24], "second": [3, 7]},
        {"subset": [11, 15, 19, 23, 2, 6], "t": 3},
        {"subset": [10, 14, 18, 22, 1, 5], "t": 4},
        {"subset": [9, 13, 17, 21, 0, 4], "t": 3},
        {"subset": [8, 12, 16, 20, 24, 3], "t": 4},
        {"subset": [7, 11, 15, 19, 23], "t": 3},
        {"subset": [2, 6, 10, 14, 18], "t": 4},
        {"subset": [22, 1, 5, 9, 13], "t": 3},
        {"subset": [17, 21, 0, 4, 8, 12], "t": 4},
        {"first": [16, 20, 24, 3], "second": [7, 11]},
        {"subset": [15, 19, 23, 2, 6, 10], "t": 3},
        {"subset": [14, 18, 22, 1, 5, 9], "t": 4},
        {"subset": [13, 17, 21, 0, 4, 8], "t": 3},
        {"subset": [12, 16, 20, 24, 3, 7], "t": 4},
        {"subset": [11, 15, 19, 23, 2], "t": 3},
        {"subset": [6, 10, 14, 18, 22], "t": 4},
        {"subset": [1, 5, 9, 13, 17], "t": 3},
        {"subset": [21, 0, 4, 8, 12, 16], "t": 4},
        {"first": [20, 24, 3, 7], "second": [11, 15]},
        {"subset": [19, 23, 2, 6, 10, 14], "t": 3},
        {"subset": [18, 22, 1, 5, 9, 13], "t": 4},
        {"subset": [17, 21, 0, 4, 8, 12], "t": 3},
        {"subset": [16, 20, 24, 3, 7, 11], "t": 4},
        {"subset": [15, 19, 23, 2, 6], "t": 3},
        {"subset": [10, 14, 18, 22, 1], "t": 4},
        {"subset": [5, 9, 13, 17, 21], "t": 3},
        {"subset": [0, 4, 8, 12, 16, 20], "t": 4},
        {"first": [24, 3, 7, 11], "second": [15, 19]},
        {"subset": [23, 2, 6, 10, 14, 18], "t": 3},
        {"subset": [22, 1, 5, 9, 13, 17], "t": 4},
        {"subset": [21, 0, 4, 8, 12, 16], "t": 3},
        {"subset": [20, 24, 3, 7, 11, 15], "t": 4},
        {"subset": [19, 23, 2, 6, 10], "t": 3},
        {"subset": [14, 18, 22, 1, 5], "t": 4},
        {"subset": [9, 13, 17, 21, 0], "t": 3},
        {"subset": [4, 8, 12, 16, 20, 24], "t": 4},
        {"first": [3, 7, 11, 15], "second": [19, 23]},
        {"subset": [2, 6, 10, 14, 18, 22], "t": 3},
        {"subset": [1, 5, 9, 13, 17, 21], "t": 4},
        {"subset": [0, 4, 8, 12, 16, 20], "t": 3},
        {"subset": [24, 3, 7, 11, 15, 19], "t": 4},
        {"subset": [23, 2, 6, 10, 14], "t": 3},
        {"subset": [18, 22, 1, 5, 9], "t": 4},
        {"subset": [13, 17, 21, 0, 4], "t": 3},
        {"subset": [8, 12, 16, 20, 24, 3], "t": 4},
        {"first": [7, 11, 15, 19], "second": [23, 2]},
        {"subset": [6, 10, 14, 18, 22, 1], "t": 3},
        {"subset": [5, 9, 13, 17, 21, 0], "t": 4},
        {"subset": [4, 8, 12, 16, 20, 24], "t": 3},
        {"subset": [3, 7, 11, 15, 19, 23], "t": 4},
        {"subset": [2, 6, 10, 14, 18], "t": 3},
        {"subset": [22, 1, 5, 9, 13], "t": 4},
        {"subset": [17, 21, 0, 4, 8], "t": 3},
        {"subset": [12, 16, 20, 24, 3, 7], "t": 4},
        {"first": [11, 15, 19, 23], "second": [2, 6]},
        {"subset": [10, 14, 18, 22, 1, 5], "t": 3},
        {"subset": [9, 13, 17, 21, 0, 4], "t": 4},
        {"subset": [8, 12, 16, 20, 24, 3], "t": 3},
        {"subset": [7, 11, 15, 19, 23, 2], "t": 4},
        {"subset": [6, 10, 14, 18, 22], "t": 3},
        {"subset": [1, 5, 9, 13, 17], "t": 4},
        {"subset": [21, 0, 4, 8, 12], "t": 3},
        {"subset": [16, 20, 24, 3, 7, 11], "t": 4},
        {"first": [15, 19, 23, 2], "second": [6, 10]},
        {"subset": [14, 18, 22, 1, 5, 9], "t": 3},
        {"subset": [13, 17, 21, 0, 4, 8], "t": 4},
        {"subset": [12, 16, 20, 24, 3, 7], "t": 3},
        {"subset": [11, 15, 19, 23, 2, 6], "t": 4},
        {"subset": [10, 14, 18, 22, 1], "t": 3},
        {"subset": [5, 9, 13, 17, 21], "t": 4},
        {"subset": [0, 4, 8, 12, 16], "t": 3},
        {"subset": [20, 24, 3, 7, 11, 15], "t": 4},
        {"first": [19, 23, 2, 6], "second": [10, 14]},
        {"subset": [18, 22, 1, 5, 9, 13], "t": 3},
        {"subset": [17, 21, 0, 4, 8, 12], "t": 4},
    ]

    for s in m:
        if "subset" in s:
            s["first"] = s["subset"][: s["t"]]

    edges = []
    for i, s_0 in enumerate(m):
        for j, s_1 in enumerate(m):
            if i < j:
                if any(x in s_1["first"] for x in s_0["first"]):
                    edges.append((i, j))

    def add_constraints(model, x, values):
        # Sample extras — replace with your real constraints
        model.AddAllDifferent([x[0], x[1], x[2]])
        model.Add(x[5] + x[6] <= x[7] + values[-1])

    model, x, y_s, y_sk, r, a = build_model(n, values, m, edges, add_additional_constraints=add_constraints)
    res = solve_model(model, x, y_s, y_sk, r, a, time_limit_sec=200)

    print("Status:", res["status"])
    if "objective" in res:
        print("Objective (k-consistent activated components):", res["objective"])
        print("x:", res["x"])
        print("winners yS:", res["yS"])
        print("component (k, representative) per winning subset:", res["component_of_subset"])
