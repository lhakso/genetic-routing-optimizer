from typing import List, Sequence, Union, Tuple
from ga_helpers import calculate_fitness, fitness_wrapper_for_rust
import ga_helpers_rs
import numpy as np
import random
import logging
import math

Numeric = Union[int, float]
DistanceMatrix = np.ndarray
Route = Sequence[int]

logger = logging.getLogger("GA_Selector_Logger")


def run_tournament(
    population_with_fitness: List[tuple[float, Route]],
    parent_count,
    tournament_size: int,
) -> List[Route]:
    if len(population_with_fitness) == 0:
        logger.warning("Population is empty. Cannot run tournament.")
        return []
    if len(population_with_fitness) == 1:
        logger.info("Only one individual in population. Returning it as the best.")
        return population_with_fitness[0]

    selected_candidates = []
    total_selected = 0

    for i in range(0, len(population_with_fitness), tournament_size):
        batch = population_with_fitness[i : i + tournament_size]
        batch_sorted = sorted(batch, key=lambda x: x[0])

        # Decide how many to take from this batch
        remaining_parents = parent_count - total_selected
        top_k = min(
            math.ceil(
                parent_count / (len(population_with_fitness) // tournament_size + 1)
            ),
            remaining_parents,
        )

        selected_candidates.extend(batch_sorted[:top_k])
        total_selected += top_k

        if total_selected >= parent_count:
            break

    # Now take the best overall from those num chosed decided by parent_count
    selected_candidates.sort(key=lambda x: x[0])
    # return [route for _, route in selected_candidates[:parent_count]]
    return selected_candidates[:parent_count]


def tournament_parent_selection(
    population: List[Route],
    elite_count: int,
    parent_selection_count: int,  # How many parents to select for breeding pool
    pool,  # Multiprocessing Pool, wont let me type hint for some reason
    tournament_size: int,
) -> Tuple[List[Route], List[Route], Route, float]:
    """
    Selects parents from the population using tournament selection.
    """
    if not population:
        logger.warning("Population is empty. Cannot select parents.")
        return []

    results = []
    fitness_scores_list = pool.map(fitness_wrapper_for_rust, population)
    population_with_fitness = list(zip(fitness_scores_list, population))
    results = run_tournament(
        population_with_fitness, parent_selection_count, tournament_size
    )
    """
    remaining_population = population.copy()
    results = []

    # 1. keep sampling until we have enough UNIQUE tuples
    while len(results) < min(
        parent_selection_count + elite_count, len(remaining_population)
    ):
        fitness, route = run_tournament(remaining_population, pool)
        remaining_population.remove(route)  # remove from population
        results.append((fitness, route))
    """
    # 2. split into elites & parents
    elite_routes = [rt for _, rt in results[:elite_count]]
    parent_routes = [
        rt for _, rt in results[elite_count : elite_count + parent_selection_count]
    ]

    best_score = results[0][0]
    best_route = elite_routes[0]

    return elite_routes, parent_routes, best_route, best_score


def select_top_elites_and_parents_mp(
    matrix: DistanceMatrix,
    population: List[Route],
    elite_count: int,
    parent_selection_count: int,  # How many parents to select for breeding pool
    pool,  # Multiprocessing Pool, wont let me type hint for some reason
) -> Tuple[List[Route], List[Route], Route, float]:
    """
    Calculates fitness for a population using a provided multiprocessing pool,
    selects elites and parents.
    """
    if not population:
        logger.warning("Population is empty. Cannot select elites or parents.")
        return [], [], [], float("inf")

    try:
        # pool.map takes a function that accepts one argument.
        # fitness_evaluator_with_matrix now fits this.
        fitness_scores_list = pool.map(fitness_wrapper_for_rust, population)
        population_with_fitness = list(zip(fitness_scores_list, population))
        logger.debug("Fitness calculation completed using multiprocessing Pool.")
    except Exception as e:
        logger.error(
            f"Error during pool.map execution: {e}. Falling back to serial processing."
        )
        population_with_fitness = []
        for route in population:
            fitness = calculate_fitness(route, matrix)
            population_with_fitness.append((fitness, route))

    if not population_with_fitness:
        logger.error("Population with fitness is empty. Cannot proceed.")
        return [], [], [], float("inf")

    sorted_population = sorted(population_with_fitness, key=lambda item: item[0])

    best_route_score: float = sorted_population[0][0]
    best_route: Route = sorted_population[0][1]

    elites: List[Route] = [
        ind[1] for ind in sorted_population[: min(elite_count, len(sorted_population))]
    ]

    # Select parents from the portion after elites
    # Ensure parent_selection_count doesn't exceed available non-elite individuals
    start_parent_idx = len(elites)
    num_candidates_for_parents = len(sorted_population) - start_parent_idx
    actual_parent_count = min(parent_selection_count, num_candidates_for_parents)

    parents: List[Route] = [
        ind[1]
        for ind in sorted_population[
            start_parent_idx : start_parent_idx + actual_parent_count
        ]
    ]

    shuffled_parents = parents.copy()
    random.shuffle(shuffled_parents)

    return elites, shuffled_parents, best_route, best_route_score
