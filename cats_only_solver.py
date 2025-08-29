import json
from itertools import combinations

from ortools.sat.python import cp_model

from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.enums.pattern import Pattern
from core.models.quilt_board import HexPosition, QuiltBoard


def build_graph_and_solve():
    # Load subsets from JSON files
    with open("data/leo_patches.json") as f:
        leo_patches = json.load(f)
    with open("data/almond_patches.json") as f:
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

    dummy_design_goals = [
        DesignGoalTiles.SIX_UNIQUE.value,
        DesignGoalTiles.THREE_PAIRS.value,
        DesignGoalTiles.TWO_TRIPLETS.value,
    ]
    board = QuiltBoard(edge_setting=EdgeTileSettings.BOARD_1, design_goal_tiles=dummy_design_goals)
    inner_pattern_restrictions: list[tuple[int, int, int]] = []
    pattern_groups: dict[Pattern, list[list[int]]] = {}
    for _, idxs in same_label_idxs.items():
        groups: dict[Pattern, list[int]] = {}
        for idx in idxs:
            subset = subsets[idx]
            for v in subset:
                pos = HexPosition.from_abs(v)
                if pos.is_edge_position():
                    pattern = board.tiles_by_pos[pos].pattern
                    groups.setdefault(pattern, []).append(idx)
                    break
        for pattern, indices in groups.items():
            pattern_groups.setdefault(pattern, []).append(indices)

        for pattern_combo in combinations(list(groups.keys()), 3):
            group1_indices = groups[pattern_combo[0]]
            group2_indices = groups[pattern_combo[1]]
            group3_indices = groups[pattern_combo[2]]

            # Find all triplets where one element comes from each group and they are disjoint
            for i1 in group1_indices:
                s_1 = set(subsets[i1])
                for i2 in group2_indices:
                    s_2 = set(subsets[i2])
                    if s_1 & s_2 != set():
                        continue
                    for i3 in group3_indices:
                        s_3 = set(subsets[i3])
                        if s_1 & s_3 != set():
                            continue
                        if s_2 & s_3 != set():
                            continue
                        sorted_indices = sorted([i1, i2, i3])
                        if sorted_indices in inner_pattern_restrictions:
                            continue
                        inner_pattern_restrictions.append((i1, i2, i3))

    outer_pattern_restrictions: list[tuple[int, int]] = []
    for _, indices_list in pattern_groups.items():
        for combo in combinations(indices_list, 2):
            for i in combo[0]:
                s_1 = set(subsets[i])
                for j in combo[1]:
                    s_2 = set(subsets[j])
                    if s_1 & s_2 != set():
                        continue
                    sorted_indices = sorted([i, j])
                    if sorted_indices in outer_pattern_restrictions:
                        continue
                    outer_pattern_restrictions.append(sorted_indices)

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

    # make sure that each cat occupy only 2 patterns
    for i, j, k in inner_pattern_restrictions:
        mdl.Add(y[i] + y[j] + y[k] <= 2)

    # make sure that no two cats occupy the same pattern
    for i, j in outer_pattern_restrictions:
        mdl.Add(y[i] + y[j] <= 1)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8  # adjust to your CPU
    solver.parameters.max_time_in_seconds = 120.0  # optional time cap
    res = solver.Solve(mdl)

    chosen = [i for i in range(m) if solver.Value(y[i]) == 1]
    return res, chosen, solver.ObjectiveValue(), subsets, weights


if __name__ == "__main__":
    result, chosen_subsets, objective_value, subsets, weights = build_graph_and_solve()

    print(f"\nSolution status: {result}")
    print(f"Objective value: {objective_value}")
    print(f"Number of chosen subsets: {len(chosen_subsets)}")

    if chosen_subsets:
        print("\nChosen subset indices:")
        for idx in chosen_subsets[:10]:  # Show first 10
            print(f"  Index {idx}: {subsets[idx]} (weight {weights[idx]})")
        if len(chosen_subsets) > 10:
            print(f"  ... and {len(chosen_subsets) - 10} more")
