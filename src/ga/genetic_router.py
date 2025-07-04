#!/Users/lukehakso/projects/kempv2/.kempv2/bin/python

import numpy as np
from typing import List, Sequence, Union, Tuple
import random
import pandas as pd
import logging
import time
import ga_helpers_rs
from multiprocessing import Pool
from ga_helpers import calculate_fitness, apply_two_opt
from visualizers import visualize_route, plot_fitness_over_generations
import os
import heapq
from gene_selectors import tournament_parent_selection, select_top_elites_and_parents_mp
from crossover import order_crossover, erx_crossover
from notifs import send_pushover_notification

# --- Logger Setup (run once at the start of your notebook/script) ---
# It's better to configure this once.
# If you run this cell multiple times, it might add multiple handlers.
logger = logging.getLogger("GA_Logger")  # Give your logger a specific name
if not logger.handlers:  # Check if handlers are already added
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- Type Aliases ---
Numeric = Union[int, float]
DistanceMatrix = np.ndarray
Route = Sequence[int]


# --- Wrapper for multiprocessing (if needed, or use partial) ---
# This version of the wrapper is if calculate_fitness is part of a class or needs other fixed args.
# If calculate_fitness is top-level and only needs route & matrix, functools.partial is cleaner.
# For this example, assuming calculate_fitness is top-level as defined above.
def run_2_opt_worker(route):
    """Worker function to run Rust 2-opt on a single route."""
    try:
        # Call the Rust function with full_run=True (as in your original code)
        optimized_route = ga_helpers_rs.apply_two_opt_from_python(route, True)
        return optimized_route
    except Exception as e:
        # It's good practice to log or handle potential errors in workers
        print(f"Error in 2-opt worker: {e}")
        return route  # Return original route on error


def python_pool_initializer(matrix_to_init_in_rust):
    """Called by each worker process when it starts."""
    global _worker_init_matrix
    # Python worker has access if needed
    _worker_init_matrix = matrix_to_init_in_rust

    # Call the Rust function to initialize its internal static matrix
    # You might need to convert numpy_matrix to a list of lists
    # if your Rust init_distance_matrix expects that and doesn't handle NumPy arrays directly.
    # Or, if using PyO3's NumPy features, you might pass it directly.
    matrix_for_rust = (
        matrix_to_init_in_rust.tolist()
    )  # Example: convert to list of lists
    ga_helpers_rs.init_distance_matrix(matrix_for_rust)
    print(f"Worker {os.getpid()} initialized Rust matrix.")


# --- GA Core Functions (Assume these are defined as in your notebook) ---
def create_initial_population(
    num_points: int, population_size: int = 100
) -> List[List[int]]:
    population: List[List[int]] = []
    points: List[int] = list(range(num_points))
    if num_points == 0:
        return []
    if num_points == 1:
        for _ in range(population_size):
            population.append([0])
        return population
    for _ in range(population_size):
        route: List[int] = random.sample(points, len(points))
        population.append(route)
    return population


def mutate(route: Sequence[int], mutation_probability: float = 0.1) -> List[int]:
    route_list = list(route)  # Work on a copy
    if random.random() < mutation_probability:
        if len(route_list) >= 2:
            i, j = random.sample(range(len(route_list)), 2)
            route_list[i], route_list[j] = route_list[j], route_list[i]
    return route_list


def have_sex_and_have_mutated_children(
    parents_list: List[Sequence[int]],
    num_offspring_needed: int,
    mutation_probability: float = 0.1,
    crossover_rate: float = 0.7,
):
    children = []
    num_available_parents = len(parents_list)
    if num_available_parents == 0:
        return []
    if num_available_parents == 1:  # Asexual reproduction if only one parent
        while len(children) < num_offspring_needed:
            children.append(mutate(parents_list[0], mutation_probability))
        return children[:num_offspring_needed]

    idx = 0
    while len(children) < num_offspring_needed:
        p1_idx = idx % num_available_parents
        p2_idx = (
            idx + 1 + random.randint(0, num_available_parents - 2)
        ) % num_available_parents  # Try to ensure different second parent
        if p1_idx == p2_idx:  # Ensure different parents if possible
            p2_idx = (p1_idx + 1) % num_available_parents

        parent1 = parents_list[p1_idx]
        parent2 = parents_list[p2_idx]

        if random.random() < crossover_rate:
            child = erx_crossover(parent1, parent2)
        else:  # Asexual reproduction (clone one parent)
            child = list(random.choice([parent1, parent2]))

        children.append(mutate(child, mutation_probability))
        idx += 1
    return children[:num_offspring_needed]


def find_route_mp(
    coords,
    matrix: DistanceMatrix,
    population_size: int = 100,
    elite_count: int = 5,
    num_generations=1000,
    mutation_prob: float = 0.05,
    parent_selection_count: int = 50,
    selection_algo_switch_gen: int = 2000,
    two_opt_start_gen: int = 1000,
    tournament_size: int = 50,
    two_opt_prob: int = 50,
):
    ga_helpers_rs.init_distance_matrix(
        matrix
    )  # Initialize the Rust matrix on main thread
    num_points = len(coords)
    if num_points == 0:
        logger.error("No coordinates provided. Cannot run GA.")
        return float("inf"), []

    population = create_initial_population(num_points, population_size)
    if not population:
        logger.error("Failed to create initial population.")
        return float("inf"), []

    best_overall_route = list(population[0])  # Initialize with a valid route
    best_overall_score = calculate_fitness(
        best_overall_route, matrix
    )  # Calculate its score
    route_scores_to_plot = []
    generations_to_plot = []
    # Create the Pool once for all generations for efficiency
    # Use context manager to ensure pool is closed properly
    num_processes = os.cpu_count() if os.cpu_count() else 2
    logger.info(f"Creating Multiprocessing Pool with {
                num_processes} processes for GA.")

    with Pool(
        processes=num_processes, initializer=python_pool_initializer, initargs=(
            matrix,)
    ) as pool:
        for generation in range(num_generations):
            if generation < selection_algo_switch_gen:
                elites, parents, current_gen_best_route, current_gen_best_score = (
                    select_top_elites_and_parents_mp(
                        matrix,
                        population,
                        elite_count,
                        parent_selection_count,
                        pool=pool,
                    )
                )
            else:
                elites, parents, current_gen_best_route, current_gen_best_score = (
                    tournament_parent_selection(
                        population,
                        elite_count,
                        parent_selection_count,
                        pool=pool,
                        tournament_size=tournament_size,
                    )
                )

            if current_gen_best_score < best_overall_score:
                best_overall_score = current_gen_best_score
                best_overall_route = list(
                    current_gen_best_route)  # Ensure it's a list
                logger.debug(
                    f"Gen {
                        generation+1}/{num_generations}: New best score: {best_overall_score:.2f}"
                )

            num_offspring_needed = population_size - len(elites)
            if (
                not parents and num_offspring_needed > 0
            ):  # Handle case where no parents might be selected if pop is too small
                logger.warning(
                    f"No parents selected in generation {
                        generation+1}, filling with mutated elites or random if elites are also few."
                )
                # Fallback: create offspring from mutated elites or new random individuals
                filler_source = (
                    elites if elites else [random.sample(
                        range(num_points), num_points)]
                )  # Use a random route if no elites
                children = [
                    mutate(list(random.choice(filler_source)), mutation_prob)
                    for _ in range(num_offspring_needed)
                ]
            elif num_offspring_needed > 0:
                children = have_sex_and_have_mutated_children(
                    parents, num_offspring_needed, mutation_prob
                )
            else:
                children = []

            if children and (generation > two_opt_start_gen):
                # Ensure two_opt_prob is a float (e.g., 0.1 for 10%)
                # Your signature had int = 50, which might be an error.
                # Assuming it should be a float probability here.
                # If it's a fixed number, just use that number.

                # Make sure two_opt_prob is a float (e.g., 0.1 for 10%)
                # If it was an int 50 meaning 50%, change it to 0.5

                two_opt_count = int(len(children) * two_opt_prob)
                if two_opt_count > 0:  # Only run if we need to optimize at least one

                    # Select indices of children to optimize
                    if generation == 1:
                        sample_indices = range(
                            len(children)
                        )  # Optimize all in first gen after random selection
                    else:
                        sample_indices = random.sample(
                            range(len(children)), two_opt_count
                        )

                    # Create a list of the actual children to process
                    children_to_process = [children[idx]
                                           for idx in sample_indices]

                    logger.info(
                        f"Running 2-opt with rust in parallel on {
                            len(children_to_process)} children..."
                    )

                    # Call pool.map - this blocks until all results are back
                    optimized_subset = pool.map(
                        run_2_opt_worker, children_to_process)

                    # Update the original children list with the results
                    # Use zip to match the original index with its optimized version
                    for original_idx, optimized_route in zip(
                        sample_indices, optimized_subset
                    ):
                        children[original_idx] = optimized_route

                    logger.info("Parallel 2-opt complete.")

            # --- Your special elite processing (keep this sequential or parallelize separately if needed) ---
            if (
                generation
                == num_generations - 2  # TODO not running this now normally is 2
            ):  # Your original logic was `num_generations - generation == 2`
                logger.info("Running 3-opt on elite")
                processed_elites = []
                for elite in elites:
                    # You could potentially run this part in parallel too,
                    # but since it's only a few elites, sequential is often fine.
                    new_elite = ga_helpers_rs.apply_three_opt_from_python(
                        elite)
                    processed_elites.append(new_elite)
                elites = processed_elites

            population = elites + children

            if ((generation + 1) / num_generations) in (0.25, 0.5, 0.75):
                send_pushover_notification(
                    "GA Checkpoint",
                    f" Generation: {generation +
                                    1}\nBest Score: {best_overall_score}",
                )

            if (  # this if elif statement is to avoid logging every generation unless under 100 gens
                num_generations > 100
                and (generation + 1) % (num_generations // 100) == 0
            ):  # Log progress
                logger.info(
                    f"--- Generation {generation + 1} complete. Best score so far: {
                        best_overall_score:.2f} ---"
                )
                route_scores_to_plot.append(best_overall_score)
                generations_to_plot.append(generation + 1)

            elif num_generations <= 100:
                logger.info(
                    f"--- Generation {generation + 1} complete. Best score so far: {
                        best_overall_score:.2f} ---"
                )
                route_scores_to_plot.append(best_overall_score)
                generations_to_plot.append(generation + 1)

    if len(best_overall_route) != len(set(best_overall_route)):
        logger.warning(
            "WARNING: Best route is not a valid permutation of all points!")

    logger.info(f"Finished. Best score: {best_overall_score:.2f}")
    return (
        best_overall_score,
        best_overall_route,
        route_scores_to_plot,
        generations_to_plot,
    )


# --- Main execution block (IMPORTANT for multiprocessing) ---
if __name__ == "__main__":
    # This ensures that the code below only runs when the script is executed directly,
    # not when it's imported by child processes spawned by multiprocessing.Pool.

    # Configure the main logger for the script
    # This basicConfig should ideally be called only once in the entire application.
    # If GA_Logger was already configured above, this might reconfigure or add handlers.
    # For a script, it's fine here.
    # logging.basicConfig(level=logging.INFO,
    # format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')
    # Use the logger instance defined at the top of the script
    start_time = time.time()
    logger.info("Script execution started under if __name__ == '__main__'.")

    BASE_DIR = os.path.dirname(__file__)
    # coords_path = os.path.join(BASE_DIR, "coordinates.csv")
    # matrix_path = os.path.join(BASE_DIR, "distance_matrix.npy")
    coords_path = os.path.join(BASE_DIR, "cluster_coords.csv")
    matrix_path = os.path.join(BASE_DIR, "matrix_cluster.npy")

    # Load data
    try:
        df = pd.read_csv(
            coords_path
        )  # Ensure this file is in the same directory or provide full path
        coords_list = list(zip(df["0"], df["1"]))
        matrix_loaded = np.load(matrix_path)  # Ensure this file is present
        logger.info(
            f"Data loaded: {len(coords_list)} coordinates, matrix shape {
                matrix_loaded.shape}"
        )
    except FileNotFoundError as e:
        logger.error(
            f"Error loading data: {
                e}. Please ensure 'coordinates.csv' and 'distance_matrix.npy' are present."
        )
        exit()  # Exit if data can't be loaded
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data loading: {e}", exc_info=True
        )
        exit()

    # GA Parameters
    POPULATION_SIZE = 500
    ELITE_COUNT = 3
    PARENT_SELECTION_COUNT = 500
    NUM_GENERATIONS = 5
    MUTATION_PROBABILITY = 0.05
    SWITCH_TO_TOURNAMENT_GEN = 0
    TWO_OPT_START_GEN = 0
    TOURNAMENT_SIZE = 3
    TWO_OPT_PROB = 0.05

    logger.info(
        f"Starting GA with Population: {POPULATION_SIZE}, Generations: {
            NUM_GENERATIONS}, Mutation: {MUTATION_PROBABILITY}"
    )

    best_score, best_rt, route_scores_to_plot, generations_to_plot = find_route_mp(
        coords_list,
        matrix_loaded,
        population_size=POPULATION_SIZE,
        elite_count=ELITE_COUNT,
        num_generations=NUM_GENERATIONS,
        mutation_prob=MUTATION_PROBABILITY,
        parent_selection_count=PARENT_SELECTION_COUNT,
        selection_algo_switch_gen=SWITCH_TO_TOURNAMENT_GEN,
        two_opt_start_gen=TWO_OPT_START_GEN,
        tournament_size=TOURNAMENT_SIZE,
        two_opt_prob=TWO_OPT_PROB,
    )

    logger.info(f"Multiprocessing GA finished.")
    logger.info(f"Best route score found: {best_score:.2f} meters")
    # logger.info(f"Best route (indices): {best_rt}")

    end_time = time.time()
    logger.info(
        f"Algorithm finished in {
            end_time - start_time:.2f} seconds with {NUM_GENERATIONS} generations."
    )

    pushover_title = "GA Finished"
    pushover_msg = f"""Best route score found: {best_score:.2f} meters, Generations: {NUM_GENERATIONS},
    Time: {end_time - start_time:.2f} seconds,
    Switch to tournament: {SWITCH_TO_TOURNAMENT_GEN},
    two_opt_start_gen: {TWO_OPT_START_GEN},
    Population: {POPULATION_SIZE},
    Mutation: {MUTATION_PROBABILITY},
    Elites: {ELITE_COUNT},
    Parents: {PARENT_SELECTION_COUNT}"""

    send_pushover_notification(pushover_title, pushover_msg)
    plot_fitness_over_generations(
        generations_to_plot,
        route_scores_to_plot,
        NUM_GENERATIONS,
        POPULATION_SIZE,
        MUTATION_PROBABILITY,
        "ERX",
        ELITE_COUNT,
        PARENT_SELECTION_COUNT,
        SWITCH_TO_TOURNAMENT_GEN,
        TWO_OPT_START_GEN,
        int(best_score),
        TOURNAMENT_SIZE,
        TWO_OPT_PROB,
    )

    if best_rt and coords_list:
        # from IPython.display import display # Needed if running in a script and want to auto-open
        route_map_viz = visualize_route(best_rt, tuple(coords_list))
        if route_map_viz:
            map_file = os.path.join(BASE_DIR, "best_route_map_mp.html")
            route_map_viz.save(map_file)
            logger.info(f"Best route map saved to {map_file}")
