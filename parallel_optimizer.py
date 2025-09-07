#!/usr/bin/env python3
"""
Parallel optimization runner for Calico solver configurations.

This module provides functionality to run multiple optimization configurations
in parallel, reading from JSON files and generating individual output files
for each configuration.
"""

import concurrent.futures
import glob
import json
from datetime import datetime
from pathlib import Path

from config_models import OptimizationConfiguration
from core.enums.edge_tile_settings import EdgeTileSettings
from solvers.combined_solver import run_optimization_for_config


def run_single_configuration(config: OptimizationConfiguration, config_index: int) -> dict:
    """
    Run optimization for a single configuration.

    This runs the full 7-configuration optimization process (testing different
    missing_pattern/missing_color combinations) for the given design goals and cats.

    Args:
        config: The optimization configuration to run
        config_index: Index of this configuration in the batch (for logging)

    Returns:
        Dictionary containing the best result and metadata
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING CONFIGURATION {config_index + 1}")
    print(f"{'=' * 80}")
    print(f"Design Goals: {' -> '.join(config.design_goals)}")
    print(f"Cats: {', '.join(config.cats)}")
    print(f"{'=' * 80}")

    try:
        # Convert configuration to format expected by combined solver
        design_goal_tiles = config.get_design_goal_tiles()
        cat_names = tuple(config.cats)
        board_setting = EdgeTileSettings.BOARD_1  # Default board setting

        # Use the existing combined solver logic
        all_results, best_result, best_config = run_optimization_for_config(design_goal_tiles, cat_names, board_setting)

        if best_result is not None:
            # Find the log filename from the combined solver (it should have been created)
            # We can reconstruct it based on the same logic used in save_combined_result_to_log
            best_score = best_result.get("objective_value", 0)
            score_str = f"{best_score:03d}"
            board_name = best_result["board_setting"]
            design_goal_str = "_".join(best_result["design_goal_names"])
            cats_str = "_".join(best_result["cat_names"])
            log_filename = f"logs/overall_solutions/{score_str}_{board_name}_{design_goal_str}_{cats_str}.log"

            print(f"\nConfiguration {config_index + 1} results saved to: {log_filename}")

            return {
                "config_index": config_index,
                "input_config": config.dict(),
                "best_result": best_result,
                "best_config": best_config,
                "all_results": all_results,
                "log_filename": log_filename,
                "success": True,
            }
        print(f"\nConfiguration {config_index + 1}: No valid solutions found!")
        return {
            "config_index": config_index,
            "input_config": config.dict(),
            "success": False,
            "error": "No valid solutions found",
        }

    except Exception as e:
        print(f"\nConfiguration {config_index + 1} failed with exception: {e}")
        return {"config_index": config_index, "input_config": config.dict(), "success": False, "error": str(e)}


def check_existing_log_file(config: OptimizationConfiguration) -> str | None:
    """
    Check if a log file already exists for this configuration.

    Args:
        config: The optimization configuration to check

    Returns:
        Path to existing log file if found, None otherwise
    """
    # Convert config to expected format to build filename pattern
    design_goal_tiles = config.get_design_goal_tiles()
    design_goal_names = [tile.config_name for tile in design_goal_tiles]

    board_name = "BOARD_1"  # Default board setting
    design_goal_str = "_".join(design_goal_names)
    cats_str = "_".join(config.cats)

    # Look for any log file matching this pattern (with any score)
    pattern = f"logs/overall_solutions/*_{board_name}_{design_goal_str}_{cats_str}.log"
    matching_files = glob.glob(pattern)

    return matching_files[0] if matching_files else None


def filter_existing_configurations(
    configurations: list[OptimizationConfiguration],
) -> tuple[list[OptimizationConfiguration], list[tuple[OptimizationConfiguration, str]]]:
    """
    Filter out configurations that already have log files.

    Args:
        configurations: List of configurations to check

    Returns:
        Tuple of (configurations_to_run, skipped_configurations_with_files)
    """
    configurations_to_run = []
    skipped_configurations = []

    print("Checking for existing log files...")

    for i, config in enumerate(configurations):
        existing_file = check_existing_log_file(config)
        if existing_file:
            skipped_configurations.append((config, existing_file))
            print(f"  Config {i + 1}: SKIPPED - Log file exists: {existing_file}")
        else:
            configurations_to_run.append(config)
            print(f"  Config {i + 1}: WILL RUN - No existing log file found")

    return configurations_to_run, skipped_configurations


def run_parallel_optimization(json_file_path: str, max_concurrency: int = 4) -> list[dict]:
    """
    Run optimization for multiple configurations in parallel.

    Args:
        json_file_path: Path to JSON file containing list of OptimizationConfiguration objects
        max_concurrency: Maximum number of configurations to run simultaneously

    Returns:
        List of result dictionaries for each configuration
    """
    # Read and validate configurations
    print(f"Loading configurations from: {json_file_path}")

    with open(json_file_path) as f:
        data = json.load(f)

    # Parse configurations
    configurations = [OptimizationConfiguration(**config_data) for config_data in data]

    print(f"Loaded {len(configurations)} configurations")

    # Create logs directory if it doesn't exist
    Path("logs/overall_solutions").mkdir(parents=True, exist_ok=True)

    # Filter out configurations that already have log files
    configurations_to_run, skipped_configurations = filter_existing_configurations(configurations)

    print("\nSummary:")
    print(f"  Total configurations: {len(configurations)}")
    print(f"  Configurations to run: {len(configurations_to_run)}")
    print(f"  Configurations skipped (existing results): {len(skipped_configurations)}")

    if not configurations_to_run:
        print("\n✅ All configurations already have results! No work needed.")

        # Return results for skipped configurations
        results = []
        for _i, (config, log_file) in enumerate(skipped_configurations):
            results.append(
                {
                    "config_index": configurations.index(config),
                    "input_config": config.dict(),
                    "success": True,
                    "skipped": True,
                    "log_filename": log_file,
                    "reason": "Log file already exists",
                }
            )
        return results

    print(f"Max concurrency: {max_concurrency}")
    print(f"Starting parallel optimization at {datetime.now()}")

    # Run configurations in parallel (only those that need to be run)
    results = []

    # First, add results for skipped configurations
    for config, log_file in skipped_configurations:
        original_index = configurations.index(config)
        results.append(
            {
                "config_index": original_index,
                "input_config": config.dict(),
                "success": True,
                "skipped": True,
                "log_filename": log_file,
                "reason": "Log file already exists",
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Submit only configurations that need to be run
        future_to_config = {
            executor.submit(run_single_configuration, config, configurations.index(config)): (
                config,
                configurations.index(config),
            )
            for config in configurations_to_run
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            config, config_index = future_to_config[future]
            try:
                result = future.result()
                results.append(result)

                if result["success"] and not result.get("skipped", False):
                    print(f"\n✅ Configuration {config_index + 1} completed successfully")
                    print(f"   Best score: {result['best_result']['objective_value']}")
                    print(f"   Log file: {result['log_filename']}")
                else:
                    print(f"\n❌ Configuration {config_index + 1} failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"\n❌ Configuration {config_index + 1} failed with exception: {e}")
                results.append(
                    {"config_index": config_index, "input_config": config.dict(), "success": False, "error": str(e)}
                )

    # Sort results by config_index to maintain original order
    results.sort(key=lambda x: x["config_index"])

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    skipped = sum(1 for r in results if r.get("skipped", False))
    ran = successful - skipped

    print(f"\n{'=' * 80}")
    print("PARALLEL OPTIMIZATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configurations)}")
    print(f"Configurations run: {ran}")
    print(f"Configurations skipped (existing results): {skipped}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Completed at: {datetime.now()}")

    if successful > 0:
        print("\nResults:")
        for result in results:
            if result["success"]:
                config_status = "SKIPPED" if result.get("skipped", False) else "RAN"
                if result.get("skipped", False):
                    print(f"  Config {result['config_index'] + 1}: {config_status} - {result['log_filename']}")
                else:
                    score = result["best_result"]["objective_value"]
                    design_goals = " -> ".join(result["input_config"]["design_goals"])
                    cats = ", ".join(result["input_config"]["cats"])
                    print(
                        f"  Config {result['config_index'] + 1}: {config_status} - "
                        f"Score: {score} ({design_goals}, {cats})"
                    )

    print(f"{'=' * 80}")

    return results


def main():
    """Main entry point for parallel optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Calico optimization configurations in parallel")
    parser.add_argument("json_file", help="Path to JSON file containing list of configurations")
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Maximum number of configurations to run in parallel (default: 4)",
    )

    args = parser.parse_args()

    try:
        results = run_parallel_optimization(args.json_file, args.concurrency)

        # Exit code based on results
        successful = sum(1 for r in results if r["success"])
        if successful == 0:
            exit(1)  # All failed
        elif successful < len(results):
            exit(2)  # Some failed
        else:
            exit(0)  # All successful

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
