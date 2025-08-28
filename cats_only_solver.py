import json

from ortools.sat.python import cp_model

from core.models.quilt_board import HexPosition


def build_graph_and_solve():
    # Load subsets from JSON files
    with open("data/leo_patches.json") as f:
        leo_patches = json.load(f)
    with open("data/cira_patches.json") as f:
        almond_patches = json.load(f)
    with open("data/millie_patches.json") as f:
        millie_patches = json.load(f)

    # Combine subsets with their weights
    subsets = []
    weights = []

    # Leo patches (weight 11)
    for patch in leo_patches:
        subsets.append(patch)
        weights.append(11)

    # Almond patches (weight 9)
    for patch in almond_patches:
        subsets.append(patch)
        weights.append(9)

    # Millie patches (weight 3)
    for patch in millie_patches:
        subsets.append(patch)
        weights.append(3)

    print(f"Total subsets: {len(subsets)}")
    print(f"Leo patches: {len(leo_patches)} (weight 11)")
    print(f"Almond patches: {len(almond_patches)} (weight 9)")
    print(f"Millie patches: {len(millie_patches)} (weight 3)")

    # Build graph with vertices 0-48
    edges = []

    for vertex in range(49):  # 0 to 48 inclusive
        try:
            hex_pos = HexPosition.from_abs(vertex)
            if hex_pos.is_valid:
                neighbors = hex_pos.get_neighbors(filtered=True)
                for neighbor in neighbors:
                    neighbor_idx = neighbor.abs
                    if neighbor_idx <= 48:  # Only include edges to vertices 0-48
                        # Add edge (ensure we don't duplicate by using vertex < neighbor_idx)
                        if vertex < neighbor_idx:
                            edges.append((vertex, neighbor_idx))
        except Exception as e:
            print(f"Error processing vertex {vertex}: {e}")

    print(f"Built graph with {len(edges)} edges")

    # Call solve method
    return solve(edges, subsets, weights)


def solve(vertex_edges, subsets, weights):
    # subsets: list of sets of vertex ids (0..48)
    # weights: list of ints/floats; each in {a,b,c}
    # vertex_edges: list of (u, v) edges on 0..48  (graph is undirected)

    vars_count = 49
    m = len(subsets)

    # Bitmasks per subset
    def mask_of(subset):
        msk = 0
        for u in subset:
            msk |= 1 << u
        return msk

    masks = [mask_of(subset) for subset in subsets]

    # adjacency bitmask on vertices
    nbr = [0] * vars_count
    for u, v in vertex_edges:
        nbr[u] |= 1 << v
        nbr[v] |= 1 << u

    # per-subset neighbor-vertices mask
    adj_mask = []
    for subset in subsets:
        msk = 0
        for u in subset:
            msk |= nbr[u]
        adj_mask.append(msk)

    # subset-level adjacency if share any edge across the base graph
    # and same label
    same_label_idxs = {}
    for i, w in enumerate(weights):
        same_label_idxs.setdefault(w, []).append(i)

    adj = [set() for _ in range(m)]
    for _, idxs in same_label_idxs.items():
        for a_i in range(len(idxs)):
            i = idxs[a_i]
            for a_j in range(a_i + 1, len(idxs)):
                j = idxs[a_j]
                if (masks[j] & adj_mask[i]) == 0 and (masks[i] & adj_mask[j]) == 0:
                    continue
                if any(_v in subsets[i] for _v in subsets[j]):
                    continue
                # if (masks[j] & adjMask[i]) != 0 or (masks[i] & adjMask[j]) != 0:
                adj[i].add(j)
                adj[j].add(i)

    # enumerate forbidden triangles within each same-label partition
    triangles = []
    for _, idxs in same_label_idxs.items():
        idxs_sorted = sorted(idxs, key=lambda t: (len(adj[t]), t))
        pos = {u: i for i, u in enumerate(idxs_sorted)}
        # work with sets as ordered by pos to avoid duplicates
        neigh_sets = {u: {v for v in adj[u] if pos.get(v, -1) > pos[u]} for u in idxs_sorted}
        for i in idxs_sorted:
            neighbors = neigh_sets[i]
            for j in neighbors:
                # intersection of higher-index neighbors
                common = neighbors & neigh_sets[j]
                for k in common:
                    triangles.append((i, j, k))

    # Build CP-SAT
    mdl = cp_model.CpModel()
    y = [mdl.NewBoolVar(f"y[{i}]") for i in range(m)]
    # objective
    mdl.Maximize(sum(int(weights[i]) * y[i] for i in range(m)))

    # disjointness per vertex
    for v in range(vars_count):
        mdl.Add(sum(y[i] for i in range(m) if (masks[i] >> v) & 1) <= 1)

    # triangle cuts
    for i, j, k in triangles:
        mdl.Add(y[i] + y[j] + y[k] <= 2)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8  # adjust to your CPU
    solver.parameters.max_time_in_seconds = 120.0  # optional time cap
    res = solver.Solve(mdl)

    chosen = [i for i in range(m) if solver.Value(y[i]) == 1]
    return res, chosen, solver.ObjectiveValue()


if __name__ == "__main__":
    result, chosen_subsets, objective_value = build_graph_and_solve()

    print(f"\nSolution status: {result}")
    print(f"Objective value: {objective_value}")
    print(f"Number of chosen subsets: {len(chosen_subsets)}")

    if chosen_subsets:
        print("\nChosen subset indices:")
        for idx in chosen_subsets[:10]:  # Show first 10
            print(f"  Index {idx}")
        if len(chosen_subsets) > 10:
            print(f"  ... and {len(chosen_subsets) - 10} more")
