# File: ga_helpers.py
import numpy as np
from typing import Sequence, Union  # List can be used if you prefer
import logging  # Optional: if you want logging within this function
import ga_helpers_rs
import random

# It's good practice to have a logger for this module too
helper_logger = logging.getLogger(__name__)  # Will be 'ga_helpers'

Numeric = Union[int, float]
DistanceMatrix = np.ndarray
Route = Sequence[int]


def fitness_wrapper_for_rust(route):
    """A simple Python wrapper that calls the Rust fitness function."""
    # The Rust function now accesses the matrix internally
    return ga_helpers_rs.calculate_fitness_rust(route)


def calculate_fitness(route: Route, distance_matrix: DistanceMatrix) -> float:
    total_distance: float = 0.0
    num_points_in_route: int = len(route)
    if num_points_in_route < 1:
        return float("inf")
    for i in range(num_points_in_route):
        from_point: int = route[i]
        to_point: int = route[(i + 1) % num_points_in_route]
        try:
            distance: Numeric = distance_matrix[from_point][to_point]
            if distance == float("inf"):
                return float("inf")
            total_distance += distance
        except IndexError:
            # Using helper_logger or print for debugging in child processes
            helper_logger.debug(
                f"IndexError in fitness: from={from_point}, to={to_point}, route_len={num_points_in_route}, matrix_shape={distance_matrix.shape}"
            )
            return float("inf")
        except TypeError:
            helper_logger.debug(
                f"TypeError in fitness: from={from_point}, to={to_point}, dist_val={distance_matrix[from_point][to_point]}"
            )
            return float("inf")
    return total_distance


def two_opt_swap(route, i, k):
    """Performs a 2-opt swap on a route by reversing the segment between indices i and k."""
    # i and k are indices, not the values themselves
    # 1. take route[0] to route[i-1] and add them in order to new_route
    # 2. take route[i] to route[k] and add them in reverse order to new_route
    # 3. take route[k+1] to end and add them in order to new_route
    new_route = route[0:i]
    new_route.extend(
        reversed(route[i : k + 1])
    )  # Note: k+1 because slice goes up to, but doesn't include, the end index
    new_route.extend(route[k + 1 :])
    return new_route


def apply_two_opt(initial_route, full_run=False):
    """
    Applies the 2-opt local search to improve a given route.

    Args:
        initial_route (list): The initial sequence of items (e.g., cities).
        distance_matrix (list of lists): Matrix where distance_matrix[i][j]
                                         is the distance between item i and item j.

    Returns:
        list: The improved route after applying 2-opt.
    """
    best_route = list(initial_route)  # Make a copy
    num_nodes = len(best_route)
    start_idx = random.randint(1, len(best_route) - 1)
    best_distance = ga_helpers_rs.calculate_fitness_rust(
        best_route
    )  # directly call the Rust function
    total_runs = 2
    if not full_run:
        for i in range(
            start_idx, start_idx + total_runs
        ):  # Start from 1 as node 0 is often a fixed start/end
            for k in range(i + 1, num_nodes):
                # Consider swapping edges (i-1, i) and (k, k+1)
                # with (i-1, k) and (i, k+1)
                # This is achieved by reversing the segment route[i:k+1]
                new_route = two_opt_swap(best_route, i, k)
                new_distance = ga_helpers_rs.calculate_fitness_rust(new_route)

                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    # print(f"improved to distance {best_distance:.2f}")
            # If an improvement was made, restart the loops to ensure all possibilities are checked again
            # for the new best_route, otherwise, the current best_route is a local optimum.
    else:
        print("Full run of 2-opt")
        improved = True
        while improved:
            improved = False
            for i in range(1, num_nodes - 1):
                for k in range(i + 1, num_nodes):
                    new_route = two_opt_swap(
                        best_route, i, k
                    )  # Assuming two_opt_swap is defined
                    new_distance = ga_helpers_rs.calculate_fitness_rust(
                        new_route
                    )  # Direct Rust call
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True

    return best_route
