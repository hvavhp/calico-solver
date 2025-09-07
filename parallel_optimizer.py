#!/usr/bin/env python3
"""
Parallel optimization runner for Calico solver configurations.

This module provides functionality to run multiple optimization configurations
in parallel, reading from JSON files and generating individual output files
for each configuration.

Features two execution modes:
1. Legacy mode: Each thread handles both model building and solving sequentially
2. Separated threads mode: Dedicated modeling threads (I/O bound) and execution
   threads (CPU bound) for better resource utilization

Usage:
    # New separated threads approach (recommended)
    python parallel_optimizer.py configs.json --separated-threads -m 30 -e 5

    # Legacy approach (for compatibility)
    python parallel_optimizer.py configs.json -c 5

The separated threads approach typically provides 2-5x performance improvement
by utilizing CPU cores more efficiently during the optimization process.
"""

import concurrent.futures
import glob
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config_models import OptimizationConfiguration
from core.enums.edge_tile_settings import EdgeTileSettings
from solvers.combined_solver import (
    build_combined_model,
    run_optimization_for_config,
    solve_combined_model,
)


class SharedModelStore:
    """Thread-safe storage for built models, shared between modeling and execution threads."""

    def __init__(self):
        self.models: dict[tuple[str, int], any] = {}
        self.lock = threading.Lock()
        self.requested_models: set[tuple[str, int]] = set()

    def get_config_key(self, config: OptimizationConfiguration, board_setting: EdgeTileSettings) -> str:
        """Create a unique key for a configuration."""
        design_goals_str = "_".join(config.design_goals)
        cats_str = "_".join(sorted(config.cats))
        return f"{design_goals_str}|{cats_str}|{board_setting.value}"

    def store_model(self, config_key: str, setting_index: int, model_components):
        """Store a built model."""
        with self.lock:
            key = (config_key, setting_index)
            self.models[key] = model_components
            print(f"📦 Model stored: {config_key[:30]}... setting {setting_index}")

    def get_model(self, config_key: str, setting_index: int) -> Any | None:
        """Get a model if available, otherwise return None."""
        with self.lock:
            key = (config_key, setting_index)
            return self.models.get(key)

    def remove_model(self, config_key: str, setting_index: int):
        """Remove a model from storage to free memory."""
        with self.lock:
            key = (config_key, setting_index)
            if key in self.models:
                del self.models[key]
                print(f"🗑️  Model removed: {config_key[:30]}... setting {setting_index}")

    def request_model(self, config_key: str, setting_index: int):
        """Mark a model as requested by an execution thread."""
        with self.lock:
            self.requested_models.add((config_key, setting_index))

    def get_requested_models(self) -> set[tuple[str, int]]:
        """Get the set of requested models."""
        with self.lock:
            return self.requested_models.copy()

    def clear_request(self, config_key: str, setting_index: int):
        """Clear a model request."""
        with self.lock:
            self.requested_models.discard((config_key, setting_index))


def get_setting_configurations():
    """Get the 7 different setting configurations."""
    return [
        {"missing_pattern": None, "missing_color": None, "name": "No missing (complete)"},
        {"missing_pattern": 1, "missing_color": None, "name": "Missing pattern for goal 1"},
        {"missing_pattern": 2, "missing_color": None, "name": "Missing pattern for goal 2"},
        {"missing_pattern": 3, "missing_color": None, "name": "Missing pattern for goal 3"},
        {"missing_pattern": None, "missing_color": 1, "name": "Missing color for goal 1"},
        {"missing_pattern": None, "missing_color": 2, "name": "Missing color for goal 2"},
        {"missing_pattern": None, "missing_color": 3, "name": "Missing color for goal 3"},
    ]


def modeling_thread_worker(
    model_store: SharedModelStore,
    modeling_jobs: list[tuple[OptimizationConfiguration, int, int]],  # (config, config_index, setting_index)
    worker_id: int,
):
    """Worker function for modeling threads that build individual models."""
    print(f"🔨 Modeling thread {worker_id} started")

    setting_configs = get_setting_configurations()

    for config, config_index, setting_index in modeling_jobs:
        try:
            # Convert configuration to format expected by combined solver
            design_goal_tiles = config.get_design_goal_tiles()
            cat_names = tuple(config.cats)
            board_setting = EdgeTileSettings.BOARD_1

            config_key = model_store.get_config_key(config, board_setting)
            setting_config = setting_configs[setting_index]

            # Check if model already exists (another thread might have built it)
            if model_store.get_model(config_key, setting_index) is not None:
                print(
                    f"🔨 Thread {worker_id}: Model already exists for config {config_index + 1}, "
                    f"setting {setting_index}"
                )
                continue

            print(
                f"🔨 Thread {worker_id}: Building {setting_config['name']} for config {config_index + 1}. "
                f"Config: {config.model_dump_json()}"
            )

            # Build the model
            combined_components = build_combined_model(
                design_goal_tiles=design_goal_tiles,
                cat_names=cat_names,
                board_setting=board_setting,
                cats_weight=1.0,
                buttons_weight=3.0,
                time_limit_s=300.0,
                missing_pattern=setting_config["missing_pattern"],
                missing_color=setting_config["missing_color"],
            )

            # Store the model
            model_store.store_model(
                config_key,
                setting_index,
                {"combined_components": combined_components, "setting_config": setting_config},
            )

        except Exception as e:
            print(f"❌ Modeling thread {worker_id} error for config {config_index + 1}, setting {setting_index}: {e}")
            continue

    print(f"🔨 Modeling thread {worker_id} finished")


def execution_thread_worker(
    model_store: SharedModelStore, configs_to_execute: list[tuple[OptimizationConfiguration, int]], worker_id: int
) -> list[dict]:
    """Worker function for execution threads that solve models."""
    print(f"⚡ Execution thread {worker_id} started")
    results = []

    for config, config_index in configs_to_execute:
        try:
            config_title = f"{config.design_goals} - {config.cats}"
            print(f"⚡ Thread {worker_id}: Processing config {config_index + 1}, {config_title}")

            # Convert configuration to format expected by combined solver
            board_setting = EdgeTileSettings.BOARD_1

            config_key = model_store.get_config_key(config, board_setting)
            setting_configs = get_setting_configurations()

            all_results = []
            best_result = None
            best_config = None
            best_score = -1
            skip_missing_color = False

            # Process each setting
            for setting_index, setting_config in enumerate(setting_configs):
                # Skip missing color configurations if we already found optimal solution
                if skip_missing_color and setting_config["missing_color"] is not None:
                    print(f"⚡ Thread {worker_id}: Skipping {setting_config['name']} (optimal found)")
                    continue

                # Request the model and wait for it to be available
                model_store.request_model(config_key, setting_index)

                model_data = None
                wait_count = 0
                max_wait_seconds = 600  # 10 minutes max wait

                while model_data is None and wait_count < max_wait_seconds:
                    model_data = model_store.get_model(config_key, setting_index)
                    if model_data is None:
                        time.sleep(1)
                        wait_count += 1
                        if wait_count % 20 == 0:  # Log every 30 seconds
                            print(
                                f"⚡ Thread {worker_id}: Still waiting for {setting_config['name']} setting of "
                                f"config {config_index + 1}, {config_title}..."
                            )

                if model_data is None:
                    print(f"❌ Thread {worker_id}: Timeout waiting for {setting_config['name']} model")
                    continue

                # Clear the request
                model_store.clear_request(config_key, setting_index)

                # Extract model components
                combined_components = model_data["combined_components"]
                setting_config = model_data["setting_config"]

                print(
                    f"⚡ Thread {worker_id}: Solving {setting_config['name']} setting of config {config_index + 1}, "
                    f"{config_title}"
                )

                # Solve the model
                result = solve_combined_model(
                    combined_components,
                    time_limit_sec=300.0,
                    workers=8,
                    missing_pattern=setting_config["missing_pattern"],
                    missing_color=setting_config["missing_color"],
                )

                # Add configuration info to result
                result["config_name"] = setting_config["name"]
                result["missing_pattern"] = setting_config["missing_pattern"]
                result["missing_color"] = setting_config["missing_color"]

                all_results.append(result)

                # Check if this is the best result so far
                if result.get("objective_value") is not None:
                    current_score = result["objective_value"]
                    print(
                        f"⚡ Thread {worker_id}: Score {current_score} for {setting_config['name']}, "
                        f"config {config_index + 1}, {config_title}"
                    )

                    if current_score > best_score:
                        best_score = current_score
                        best_result = result
                        best_config = setting_config
                        print(
                            f"⚡ Thread {worker_id}: NEW BEST SCORE {current_score}!, config {config_index + 1}, "
                            f"{config_title}"
                        )

                    # # Check for optimal solution with max button score
                    # if (setting_index == 0 and result.get("status") == "OPTIMAL"
                    #     and result.get("buttons_objective_value", 0) >= 11):
                    #     skip_missing_color = True
                    #     print(f"⚡ Thread {worker_id}: Optimal solution found, skipping missing color configs")

                # Remove model from store to free memory
                model_store.remove_model(config_key, setting_index)

            # Create result for this configuration
            if best_result is not None:
                # Construct log filename
                score_str = f"{best_score:03d}"
                board_name = best_result["board_setting"]
                design_goal_str = "_".join(best_result["design_goal_names"])
                cats_str = "_".join(best_result["cat_names"])
                log_filename = f"logs/overall_solutions/{score_str}_{board_name}_{design_goal_str}_{cats_str}.log"

                result_dict = {
                    "config_index": config_index,
                    "input_config": config.dict(),
                    "best_result": best_result,
                    "best_config": best_config,
                    "all_results": all_results,
                    "log_filename": log_filename,
                    "success": True,
                }
                print(f"⚡ Thread {worker_id}: Config {config_index + 1} completed with score {best_score}")
            else:
                result_dict = {
                    "config_index": config_index,
                    "input_config": config.dict(),
                    "success": False,
                    "error": "No valid solutions found",
                }
                print(f"⚡ Thread {worker_id}: Config {config_index + 1} found no solutions")

            results.append(result_dict)

        except Exception as e:
            print(f"❌ Execution thread {worker_id} error for config {config_index + 1}: {e}")
            results.append(
                {
                    "config_index": config_index,
                    "input_config": config.dict(),
                    "success": False,
                    "error": str(e),
                }
            )

    print(f"⚡ Execution thread {worker_id} finished")
    return results


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


def run_parallel_optimization_with_separated_threads(
    json_file_path: str, modeling_workers: int = 30, execution_workers: int = 5
) -> list[dict]:
    """
    Run optimization with separate modeling and execution thread pools.

    Args:
        json_file_path: Path to JSON file containing list of OptimizationConfiguration objects
        modeling_workers: Number of threads to use for building models
        execution_workers: Number of threads to use for solving models

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

    print(f"Modeling workers: {modeling_workers}")
    print(f"Execution workers: {execution_workers}")
    print(f"Starting parallel optimization with separated threads at {datetime.now()}")

    # Initialize shared model store
    model_store = SharedModelStore()

    # Prepare configuration lists with indices for execution threads
    configs_with_indices = [(config, configurations.index(config)) for config in configurations_to_run]

    # Create individual modeling jobs (config, config_index, setting_index)
    modeling_jobs = []
    for config, config_index in configs_with_indices:
        for setting_index in range(7):  # 7 settings (0-6)
            modeling_jobs.append((config, config_index, setting_index))

    print(f"Created {len(modeling_jobs)} modeling jobs ({len(configs_with_indices)} configs × 7 settings)")

    # Distribute modeling jobs using round-robin (horizontal) to prioritize early jobs
    modeling_work_chunks = [[] for _ in range(modeling_workers)]
    for i, job in enumerate(modeling_jobs):
        worker_index = i % modeling_workers
        modeling_work_chunks[worker_index].append(job)

    # Remove empty chunks
    modeling_work_chunks = [chunk for chunk in modeling_work_chunks if chunk]

    # Distribute execution jobs using round-robin (horizontal) to prioritize early configs
    execution_work_chunks = [[] for _ in range(execution_workers)]
    for i, config_job in enumerate(configs_with_indices):
        worker_index = i % execution_workers
        execution_work_chunks[worker_index].append(config_job)

    # Remove empty chunks
    execution_work_chunks = [chunk for chunk in execution_work_chunks if chunk]

    all_results = []

    # First, add results for skipped configurations
    for config, log_file in skipped_configurations:
        original_index = configurations.index(config)
        all_results.append(
            {
                "config_index": original_index,
                "input_config": config.dict(),
                "success": True,
                "skipped": True,
                "log_filename": log_file,
                "reason": "Log file already exists",
            }
        )

    # Start modeling and execution threads concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=modeling_workers + execution_workers) as executor:
        # Submit modeling workers
        modeling_futures = []
        for i, work_chunk in enumerate(modeling_work_chunks):
            future = executor.submit(modeling_thread_worker, model_store, work_chunk, i + 1)
            modeling_futures.append(future)

        # Submit execution workers
        execution_futures = []
        for i, work_chunk in enumerate(execution_work_chunks):
            future = executor.submit(execution_thread_worker, model_store, work_chunk, i + 1)
            execution_futures.append(future)

        # Wait for all execution threads to complete (modeling threads may finish earlier)
        print("Waiting for execution threads to complete...")
        execution_results = []
        for future in concurrent.futures.as_completed(execution_futures):
            try:
                thread_results = future.result()
                execution_results.extend(thread_results)
            except Exception as e:
                print(f"❌ Execution thread failed with exception: {e}")

        # Wait for modeling threads to finish (cleanup)
        print("Waiting for modeling threads to finish...")
        for future in concurrent.futures.as_completed(modeling_futures):
            try:
                future.result()  # Just wait for completion
            except Exception as e:
                print(f"❌ Modeling thread failed with exception: {e}")

    # Add execution results to all results
    all_results.extend(execution_results)

    # Sort results by config_index to maintain original order
    all_results.sort(key=lambda x: x["config_index"])

    # Print summary
    successful = sum(1 for r in all_results if r["success"])
    failed = len(all_results) - successful
    skipped = sum(1 for r in all_results if r.get("skipped", False))
    ran = successful - skipped

    print(f"\n{'=' * 80}")
    print("PARALLEL OPTIMIZATION SUMMARY (SEPARATED THREADS)")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configurations)}")
    print(f"Configurations run: {ran}")
    print(f"Configurations skipped (existing results): {skipped}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Modeling workers used: {modeling_workers}")
    print(f"Execution workers used: {execution_workers}")
    print(f"Completed at: {datetime.now()}")

    if successful > 0:
        print("\nResults:")
        for result in all_results:
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

    return all_results


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
        help="Maximum number of configurations to run in parallel (default: 4, only used in legacy mode)",
    )
    parser.add_argument(
        "--separated-threads",
        action="store_true",
        help="Use separated modeling and execution threads (recommended for better CPU utilization)",
    )
    parser.add_argument(
        "--modeling-workers",
        "-m",
        type=int,
        default=30,
        help="Number of modeling threads (default: 30, only used with --separated-threads)",
    )
    parser.add_argument(
        "--execution-workers",
        "-e",
        type=int,
        default=2,
        help="Number of execution threads (default: 2, only used with --separated-threads)",
    )

    args = parser.parse_args()

    try:
        if args.separated_threads:
            print("🚀 Using separated modeling and execution threads")
            results = run_parallel_optimization_with_separated_threads(
                args.json_file, args.modeling_workers, args.execution_workers
            )
        else:
            print("🔄 Using legacy combined threading approach")
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
