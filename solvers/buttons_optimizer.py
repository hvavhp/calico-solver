from ortools.sat.python import cp_model


def build_model(n, values, m, add_additional_constraints=None):
    """
    n: number of variables x_0..x_{n-1}
    values: list of allowed integer values [v1,..,vK] (not necessarily contiguous)
    m: list of subsets; each item can be:
       - {"first": [i,...], "second": [j,...]} indices are 0-based
       - OR {"subset": [idx0, idx1, ...], "t": t} meaning first=subset[:t], second=subset[t:]
    add_additional_constraints: optional callback (model, x, values) -> None to add your constraints
    """
    k = len(values)
    model = cp_model.CpModel()

    # 1) Decision vars x_i in the given finite set of values
    x = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(values), f"x[{i}]") for i in range(n)]

    # 2) Equality literals eq[(i,k)] <=> (x[i] == values[k])
    eq = {}
    for i in range(n):
        for k, val in enumerate(values):
            b = model.NewBoolVar(f"eq_x[{i}]_v{k}")
            model.Add(x[i] == val).OnlyEnforceIf(b)
            model.Add(x[i] != val).OnlyEnforceIf(b.Not())
            eq[(i, k)] = b

    # 3) Parse m into (first_indices, second_indices)
    parsed_m = []
    for _m_idx, item in enumerate(m):
        if "first" in item and "second" in item:
            first = item["first"]
            second = item["second"]
        else:
            subset = item["subset"]
            t = item["t"]
            assert 1 <= t <= len(subset), "Assumption: t>=1 and t<=len(subset)"
            first = subset[:t]
            second = subset[t:]
        parsed_m.append((first, second))

    # 4) Build y_{S,k} and y_S with reified ANDs
    y_s_list = []
    for s_idx, (first, second) in enumerate(parsed_m):
        y_s = model.NewBoolVar(f"y_subset[{s_idx}]")
        y_sk_list = []
        for k_val in range(k):
            # Conditions: all i in first equal v_k, and all j in second NOT equal v_k
            conds = [eq[(i, k_val)] for i in first] + [eq[(j, k_val)].Not() for j in second]

            y_sk = model.NewBoolVar(f"y_subset[{s_idx}]_val{k_val}")

            # Enforce y_sk <=> AND(conds)
            # (1) y_sk => all conds
            for lit in conds:
                model.AddImplication(y_sk, lit)
            # (2) all conds => y_sk  ==  OR(not conds, y_sk)
            model.AddBoolOr([c.Not() for c in conds] + [y_sk])
            # (3) (optional but helpful) also add: If y_sk is true then AND(conds) must hold:
            model.AddBoolAnd(conds).OnlyEnforceIf(y_sk)

            y_sk_list.append(y_sk)

        # Because first is non-empty, at most one k can be true.
        # Tie y_s to the disjunction: equality is safe and convenient.
        model.Add(sum(y_sk_list) == y_s)
        y_s_list.append(y_s)

    # 5) Objective: maximize sum of y_s
    model.Maximize(sum(y_s_list))

    # 6) Hook for your other constraints on x
    if add_additional_constraints is not None:
        add_additional_constraints(model, x, values)

    return model, x, y_s_list


def solve_model(model, x, y_s_list, time_limit_sec=None):
    solver = cp_model.CpSolver()
    if time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8  # adjust if you like
    status = solver.Solve(model)

    res = {
        "status": solver.StatusName(status),
        "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "x": [solver.Value(v) for v in x] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "y": [solver.Value(y) for y in y_s_list] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
    }
    return res


# ---------------------------
# Example usage (replace with your data)
# ---------------------------
if __name__ == "__main__":
    n = 25
    values = [10, 20, 30, 40, 50, 60]  # your v1..v6

    # Generate 100 subsets with larger sizes and larger first groups
    M = [
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

    def add_constraints(model, x, values):
        # A few sample "original" constraints; replace with your real ones.
        # Example: x[0] != x[1]
        model.Add(x[0] != x[1])
        # Example: all-different on a subset
        model.AddAllDifferent([x[2], x[3], x[4]])
        # Example: linear relation via actual numeric values
        model.Add(x[5] + x[6] <= x[7] + values[-1])

    model, x, y_s = build_model(n, values, M, add_constraints)
    result = solve_model(model, x, y_s, time_limit_sec=10)

    print("Status:", result["status"])
    if result["objective"] is not None:
        print("Objective:", result["objective"])
        print("x:", result["x"])
        print("subset scores y:", result["y"])
