#!/usr/bin/env python3
"""
Comprehensive Solver Runner for Calico Solver Project

This script provides a unified interface to run all available solvers.
Select which solver to run from an interactive menu.
"""

import argparse

from core.enums.design_goal import DesignGoalTiles
from core.enums.edge_tile_settings import EdgeTileSettings
from core.models.quilt_board import QuiltBoard
from parallel_optimizer import run_parallel_optimization
from solvers.all_solver import main as all_solver_main
from solvers.buttons_solver import main as buttons_solver_main
from solvers.cats_modeler import main as cats_modeler_main
from solvers.combined_solver import main as combined_solver_main
from solvers.new_button_solver import main as new_button_solver_main


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print(" 🧩 CALICO SOLVER - Comprehensive Solver Runner")
    print("=" * 60)
    print()


def print_solver_menu():
    """Display available solvers menu."""
    print("Available Solvers:")
    print("─" * 30)
    print("1. Buttons Optimizer")
    print("   - General optimization solver using CP-SAT")
    print("   - Maximizes subset scores with constraints")
    print()
    print("2. Design Goals Solver (Three Patterns)")
    print("   - Solves three pattern problems")
    print("   - Options: single, all, or unique solutions")
    print()
    print("3. Full Design Goals Solver")
    print("   - Combined patterns with pair capacity constraints")
    print("   - Solves primed/unprimed variable systems")
    print()
    print("4. New Design Goals Solver")
    print("   - Enhanced version of combined patterns solver")
    print("   - Includes quilt board integration")
    print()
    print("5. Component Solver")
    print("   - Component counting with edge constraints")
    print("   - Maximizes connected components")
    print()
    print("6. K-Consistent Component Solver")
    print("   - Enhanced component solver")
    print("   - K-consistent component counting")
    print()
    print("7. Quilt Board Pretty Print Demo")
    print("   - Demonstrates hexagonal board visualization")
    print("   - Shows edge tiles, design goals, and layout")
    print()
    print("8. Buttons Solver")
    print("   - Buttons solver")
    print("   - Maximizes buttons")
    print()
    print("9. Cats Modeler")
    print("   - Cat-based patch selection and optimization")
    print("   - Combines design goals with cat patch scoring")
    print()
    print("10. Combined Solver")
    print("   - Integrates design goals, cats, and buttons constraints")
    print("   - Comprehensive optimization with all components")
    print()
    print("11. All Solver")
    print("   - All solver comprehensive")
    print("   - Advanced optimization features")
    print()
    print("12. Parallel Optimizer")
    print("   - Run multiple configurations from JSON file in parallel")
    print("   - Process multiple design goals/cats combinations concurrently")
    print()
    print("0. Exit")
    print("─" * 30)


def run_buttons_optimizer():
    """Run the buttons optimizer solver."""
    print("\n🚀 Running Buttons Optimizer...")
    print("─" * 40)

    try:
        from solvers.buttons_optimizer import build_model, solve_model

        # Default parameters (you can modify these as needed)
        n = 25
        values = [10, 20, 30, 40, 50, 60]

        # Generate sample constraint sets
        m = [
            {"subset": [0, 1, 2, 3, 4, 5], "t": 4},
            {"first": [2, 5, 8, 11], "second": [6, 7, 1]},
            {"subset": [4, 6, 0, 9, 12], "t": 3},
            {"subset": [8, 9, 10, 11, 13, 14], "t": 4},
            {"subset": [12, 13, 14, 15, 16], "t": 3},
        ]

        def add_constraints(model, x, values):
            model.Add(x[0] != x[1])
            model.AddAllDifferent([x[2], x[3], x[4]])
            model.Add(x[5] + x[6] <= x[7] + values[-1])

        print(f"Problem size: {n} variables, {len(values)} values, {len(m)} constraints")

        model, x, y_s = build_model(n, values, m, add_constraints)
        result = solve_model(model, x, y_s, time_limit_sec=10)

        print("\nResults:")
        print(f"Status: {result['status']}")
        if result["objective"] is not None:
            print(f"Objective: {result['objective']}")
            print(f"Solution x: {result['x'][:10]}..." if len(result["x"]) > 10 else f"Solution x: {result['x']}")
            print(f"Subset scores y: {result['y']}")

        return True

    except Exception as e:
        print(f"❌ Error running Buttons Optimizer: {e}")
        return False


def run_design_goals_solver():
    """Run the design goals solver (three patterns)."""
    print("\n🚀 Running Design Goals Solver (Three Patterns)...")
    print("─" * 50)

    # Ask user for solver mode
    print("Select mode:")
    print("1. Single solution")
    print("2. All solutions")
    print("3. All unique solutions (deduplicated)")

    while True:
        try:
            mode = int(input("Enter mode (1-3): "))
            if mode in [1, 2, 3]:
                break
            print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")

    try:
        from solvers.design_goals_solver import (
            solve_three_patterns,
            solve_three_patterns_all,
            solve_three_patterns_all_unique,
        )

        # Default parameters
        v = [3, 7, 11, 20, 25, 42]
        m1 = [1, 1, 1, 1, 1, 1]
        m2 = [3, 2, 1]
        m3 = [4, 1, 1]

        print(f"Values: {v}")
        print(f"Pattern 1: {m1}")
        print(f"Pattern 2: {m2}")
        print(f"Pattern 3: {m3}")
        print()

        if mode == 1:
            result = solve_three_patterns(v, m1, m2, m3)
            if result is None:
                print("❌ No feasible solution found")
            else:
                x_sol, t_sol, p_sol = result
                print("✅ Solution found:")
                print(f"X: {x_sol}")
                print(f"T: {t_sol}")
                print(f"P: {p_sol}")
                print(
                    f"Constraint checks: x3=t1? {x_sol[2] == t_sol[0]}, "
                    f"x4=t6? {x_sol[3] == t_sol[5]}, x5=p2? {x_sol[4] == p_sol[1]}"
                )

        elif mode == 2:
            solutions = solve_three_patterns_all(v, m1, m2, m3, max_solutions=50)
            if not solutions:
                print("❌ No feasible solutions found")
            else:
                print(f"✅ Found {len(solutions)} solutions")
                for idx, (x_sol, t_sol, p_sol) in enumerate(solutions[:3], 1):
                    print(f"Solution {idx}: X={x_sol}, T={t_sol}, P={p_sol}")
                if len(solutions) > 3:
                    print(f"... and {len(solutions) - 3} more solutions")

        elif mode == 3:
            solutions = solve_three_patterns_all_unique(v, m1, m2, m3, max_solutions=50000)
            if not solutions:
                print("❌ No feasible solutions found")
            else:
                print(f"✅ Found {len(solutions)} unique solutions")
                for idx, (x_sol, t_sol, p_sol) in enumerate(solutions, 1):
                    print(f"Unique solution {idx}: X={x_sol}, T={t_sol}, P={p_sol}")

        return True

    except Exception as e:
        print(f"❌ Error running Design Goals Solver: {e}")
        return False


def run_full_design_goals_solver():
    """Run the full design goals solver."""
    print("\n🚀 Running Full Design Goals Solver...")
    print("─" * 45)

    try:
        from solvers.full_design_goals_solver import solve_combined

        # Default parameters
        v = [1, 2, 3, 4, 5, 6]
        m1 = [2, 2, 2]  # X pattern: AA-BB-CC
        m2 = [3, 2, 1]  # T pattern: AAA-BB-C
        m3 = [4, 1, 1]  # P pattern: AAAA-B-C
        cap = 3

        print(f"Values: {v}")
        print(f"X pattern: {m1}")
        print(f"T pattern: {m2}")
        print(f"P pattern: {m3}")
        print(f"Pair capacity: {cap}")
        print()

        solutions = solve_combined(v, m1, m2, m3, cap=cap, time_limit_s=5.0)

        if solutions is None:
            print("❌ No feasible solutions found")
        else:
            print(f"✅ Found {len(solutions)} total feasible solutions")
            for idx, sol in enumerate(solutions[:2], 1):
                print(f"Solution {idx}:")
                for key, vals in sol.items():
                    if key != "pair_counts_total":
                        print(f"  {key}: {vals}")
            if len(solutions) > 2:
                print(f"... and {len(solutions) - 2} more solutions")

        return True

    except Exception as e:
        print(f"❌ Error running Full Design Goals Solver: {e}")
        return False


def run_new_design_goals_solver():
    """Run the new design goals solver."""
    print("\n🚀 Running New Design Goals Solver...")
    print("─" * 44)

    from solvers.restructured_design_goals_solver import solve_combined

    # Default parameters
    v = [1, 2, 3, 4, 5, 6]
    m1 = DesignGoalTiles.THREE_PAIRS.value
    m2 = DesignGoalTiles.THREE_TWO_ONE.value
    m3 = DesignGoalTiles.FOUR_TWO.value
    cap = 3

    print(f"Values: {v}")
    print(f"X pattern: {m1}")
    print(f"T pattern: {m2}")
    print(f"P pattern: {m3}")
    print(f"Pair capacity: {cap}")
    print("Note: This solver includes QuiltBoard integration")
    print()

    solutions = solve_combined(v, m1, m2, m3, cap=cap, time_limit_s=5.0)

    if solutions is None:
        print("❌ No feasible solutions found")
    else:
        print(f"✅ Found {len(solutions)} total feasible solutions")
        for idx, sol in enumerate(solutions[:2], 1):
            print(f"Solution {idx}:")
            for key, vals in sol.items():
                if key != "pair_counts_total":
                    print(f"  {key}: {vals}")
        if len(solutions) > 2:
            print(f"... and {len(solutions) - 2} more solutions")

    return True


def run_component_solver():
    """Run the component solver."""
    print("\n🚀 Running Component Solver...")
    print("─" * 35)

    try:
        from solvers.button_solver import build_model, solve_model

        # Default parameters
        n = 25
        values = [10, 20, 30, 40, 50, 60]

        # Sample constraint sets
        m = [
            {"subset": [0, 1, 2, 3, 4, 5], "t": 4},
            {"first": [2, 5, 8, 11], "second": [6, 7, 1]},
            {"subset": [4, 6, 0, 9, 12], "t": 3},
            {"subset": [8, 9, 10, 11, 13, 14], "t": 4},
            {"subset": [12, 13, 14, 15, 16], "t": 3},
        ]

        # Add first groups for edge computation
        for s in m:
            if "subset" in s:
                s["first"] = s["subset"][: s["t"]]

        # Create edges between subsets that share elements in first groups
        edges = []
        for i, s_0 in enumerate(m):
            for j, s_1 in enumerate(m):
                if i < j:
                    if any(x in s_1["first"] for x in s_0["first"]):
                        edges.append((i, j))

        def add_constraints(model, x, values):
            model.Add(x[0] != x[1])
            model.AddAllDifferent([x[2], x[3], x[4]])
            model.Add(x[5] + x[6] <= x[7] + values[-1])

        print(f"Problem size: {n} variables, {len(values)} values, {len(m)} subsets, {len(edges)} edges")

        model, x, y, r, a = build_model(n, values, m, edges, add_additional_constraints=add_constraints)
        result = solve_model(model, x, y, r, a, time_limit_sec=10)

        print("\nResults:")
        print(f"Status: {result['status']}")
        if result["objective"] is not None:
            print(f"Objective (components): {result['objective']}")
            print(f"Solution x: {result['x'][:10]}..." if len(result["x"]) > 10 else f"Solution x: {result['x']}")
            print(f"Winning subsets y: {result['y']}")
            print(
                f"Component assignment: {result['component_of_subset'][:5]}..."
                if len(result["component_of_subset"]) > 5
                else f"Component assignment: {result['component_of_subset']}"
            )

        return True

    except Exception as e:
        print(f"❌ Error running Component Solver: {e}")
        return False


def run_k_consistent_solver():
    """Run the k-consistent component solver."""
    print("\n🚀 Running K-Consistent Component Solver...")
    print("─" * 47)

    try:
        from solvers.new_button_solver import build_model, solve_model

        # Default parameters
        n = 25
        values = [10, 20, 30, 40, 50, 60]

        # Sample constraint sets
        m = [
            {"subset": [0, 1, 2, 3, 4, 5], "t": 4},
            {"first": [2, 5, 8, 11], "second": [6, 7, 1]},
            {"subset": [4, 6, 0, 9, 12], "t": 3},
            {"subset": [8, 9, 10, 11, 13, 14], "t": 4},
            {"subset": [12, 13, 14, 15, 16], "t": 3},
        ]

        # Add first groups for edge computation
        for s in m:
            if "subset" in s:
                s["first"] = s["subset"][: s["t"]]

        # Create edges between subsets that share elements in first groups
        edges = []
        for i, s_0 in enumerate(m):
            for j, s_1 in enumerate(m):
                if i < j:
                    if any(x in s_1["first"] for x in s_0["first"]):
                        edges.append((i, j))

        def add_constraints(model, x, values):
            model.AddAllDifferent([x[0], x[1], x[2]])
            model.Add(x[5] + x[6] <= x[7] + values[-1])

        print(f"Problem size: {n} variables, {len(values)} values, {len(m)} subsets, {len(edges)} edges")
        print("Note: This solver uses k-consistent component counting")

        model, x, ys, ysk, r, a = build_model(None, n, values, m, edges, add_additional_constraints=add_constraints)
        result = solve_model(model, x, ys, ysk, r, a, time_limit_sec=200)

        print("\nResults:")
        print(f"Status: {result['status']}")
        if "objective" in result and result["objective"] is not None:
            print(f"Objective (k-consistent components): {result['objective']}")
            print(f"Solution x: {result['x'][:10]}..." if len(result["x"]) > 10 else f"Solution x: {result['x']}")
            print(f"Winning subsets ys: {result['yS']}")
            print(
                f"Component (k, representative): {result['component_of_subset'][:5]}..."
                if len(result["component_of_subset"]) > 5
                else f"Component (k, representative): {result['component_of_subset']}"
            )

        return True

    except Exception as e:
        print(f"❌ Error running K-Consistent Component Solver: {e}")
        return False


def run_quilt_board_demo() -> bool:
    """Demonstrate the quilt board pretty print functionality."""
    try:
        print("\n🧩 Quilt Board Pretty Print Demonstration")
        print("=" * 50)

        print("Creating sample quilt boards with different edge tile configurations...\n")

        # Try different board configurations
        board_configs = [
            (EdgeTileSettings.BOARD_1, "Board 1 - Alternating stripe patterns"),
            (EdgeTileSettings.BOARD_2, "Board 2 - Warm/cool color distribution"),
            (EdgeTileSettings.BOARD_3, "Board 3 - Pattern diversity emphasis"),
            (EdgeTileSettings.BOARD_4, "Board 4 - Rotational symmetry"),
        ]

        for edge_setting, description in board_configs:
            print(f"\n{description}")
            print("-" * len(description))

            # Create a quilt board with the specific edge setting
            design_goals = [
                DesignGoalTiles.SIX_UNIQUE.value,
                DesignGoalTiles.THREE_PAIRS.value,
                DesignGoalTiles.TWO_TRIPLETS.value,
            ]

            board = QuiltBoard(edge_setting=edge_setting, design_goal_tiles=design_goals)

            # Print the board
            print(board.pretty_print())

            # Option to see next board
            if edge_setting != EdgeTileSettings.BOARD_4:  # Don't ask after the last one
                user_input = input("\nPress Enter to see next board configuration, or 'q' to finish demo: ").strip()
                if user_input.lower() == "q":
                    break

        print("\n✅ Quilt Board Pretty Print Demo completed successfully!")
        print("\nThe pretty_print() method shows:")
        print("- Edge tiles with their actual colors and patterns")
        print("- Design goal tiles as G1, G2, G3 at fixed positions")
        print("- Hexagonal layout with odd rows shifted right")
        print("- Empty spaces for patch tiles that haven't been placed yet")

        return True

    except Exception as e:
        print(f"❌ Error running Quilt Board Demo: {e}")
        return False


def run_parallel_optimizer():
    """Run the parallel optimizer."""
    print("\n🚀 Running Parallel Optimizer...")
    print("─" * 40)

    # Ask for JSON file path
    json_file = input(
        "Enter path to JSON configuration file (or press Enter for example_configurations.json): "
    ).strip()
    if not json_file:
        json_file = "example_configurations.json"

    # Ask for concurrency
    try:
        concurrency_input = input("Enter max concurrency (default 4): ").strip()
        concurrency = int(concurrency_input) if concurrency_input else 4
    except ValueError:
        concurrency = 4

    try:
        results = run_parallel_optimization(json_file, concurrency)

        # Print summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        print("\n✅ Parallel optimization completed!")
        print(f"Total configurations: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        return True

    except Exception as e:
        print(f"❌ Error running Parallel Optimizer: {e}")
        return False


def get_user_choice() -> int | None:
    """Get user's solver choice."""
    try:
        choice = input("Enter your choice (1-12): ").strip()
        return int(choice) if choice.isdigit() else None
    except (ValueError, KeyboardInterrupt):
        return None


def main():
    """Main runner function."""
    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Calico Solver Runner")
    parser.add_argument(
        "--solver",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Run specific solver directly (1-12)",
    )
    args = parser.parse_args()

    print_banner()

    # If solver specified via command line, run it directly
    if args.solver:
        print(f"Running solver {args.solver} directly...\n")
        solvers = {
            1: run_buttons_optimizer,
            2: run_design_goals_solver,
            3: run_full_design_goals_solver,
            4: run_new_design_goals_solver,
            5: run_component_solver,
            6: new_button_solver_main,
            7: run_quilt_board_demo,
            8: buttons_solver_main,
            9: cats_modeler_main,
            10: combined_solver_main,
            11: all_solver_main,
            12: run_parallel_optimizer,
        }
        solvers[args.solver]()

    # Interactive mode
    while True:
        print_solver_menu()
        choice = get_user_choice()

        if choice == 0:
            print("\n👋 Goodbye! Thanks for using Calico Solver!")
            break
        if choice == 1:
            run_buttons_optimizer()
        elif choice == 2:
            run_design_goals_solver()
        elif choice == 3:
            run_full_design_goals_solver()
        elif choice == 4:
            run_new_design_goals_solver()
        elif choice == 5:
            run_component_solver()
        elif choice == 6:
            new_button_solver_main()
        elif choice == 7:
            run_quilt_board_demo()
        elif choice == 8:
            buttons_solver_main()
        elif choice == 9:
            cats_modeler_main()
        elif choice == 10:
            combined_solver_main()
        elif choice == 11:
            all_solver_main()
        elif choice == 12:
            run_parallel_optimizer()
        else:
            print("❌ Invalid choice. Please enter a number from 0-12.")

        if choice != 0:
            print("\n" + "=" * 60)
            input("Press Enter to continue...")
            print("\n")


if __name__ == "__main__":
    main()
