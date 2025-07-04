from typing import Sequence, List, Dict, Set
import logging
import random

logger = logging.getLogger(__name__)


def order_crossover(parent1: Sequence[int], parent2: Sequence[int]) -> List[int]:
    size = len(parent1)
    if size == 0:
        return []
    if size != len(parent2):
        raise ValueError("Parents must be of the same length.")
    child: List[int | None] = [None] * size
    point1, point2 = sorted(random.sample(range(size), 2))
    start_index, end_index = point1, point2

    child_segment_from_parent1 = list(parent1[start_index : end_index + 1])
    for i in range(len(child_segment_from_parent1)):
        child[start_index + i] = child_segment_from_parent1[i]

    elements_in_child_segment = set(child_segment_from_parent1)
    parent2_ptr = (end_index + 1) % size
    child_fill_ptr = (end_index + 1) % size

    filled_count = len(child_segment_from_parent1)
    while filled_count < size:
        item_from_parent2 = parent2[parent2_ptr]
        if item_from_parent2 not in elements_in_child_segment:
            child[child_fill_ptr] = item_from_parent2
            child_fill_ptr = (child_fill_ptr + 1) % size
            filled_count += 1
        parent2_ptr = (parent2_ptr + 1) % size
        if (
            parent2_ptr == (end_index + 1) % size and filled_count < size
        ):  # Safety break / error condition
            # This indicates an issue, possibly non-unique elements in parents or logic error
            logger.error(
                f"Crossover error: Could not fill child. P1:{parent1}, P2:{parent2}, Child:{child}"
            )
            # Fallback: return a copy of a parent to ensure valid permutation
            return list(parent1)
    final_child: List[int] = [
        c for c in child if c is not None
    ]  # Should all be non-None
    if len(final_child) != size:  # Should not happen if logic above is perfect
        logger.error(
            f"Crossover produced child of wrong length. P1:{parent1}, Child:{final_child}"
        )
        return list(parent1)
    return final_child


def erx_crossover(parent1: Sequence[int], parent2: Sequence[int]) -> List[int]:
    """
    Performs Edge Recombination Crossover (ERX) for the Traveling Salesperson Problem.

    Args:
        parent1: The first parent tour (a sequence of city identifiers).
        parent2: The second parent tour (a sequence of city identifiers).

    Returns:
        A new child tour created by recombining edges from the parents.
        Returns a copy of parent1 as a fallback in case of errors or inability to complete.
    """
    size = len(parent1)

    if size == 0:
        return []
    if size == 1:  # Base case for a single-city tour
        return list(parent1)

    if size != len(parent2):
        logger.error("ERX Crossover error: Parents must be of the same length.")
        raise ValueError("Parents must be of the same length.")

    # Assuming parent1 and parent2 are permutations of the same set of cities.
    # Get all unique city identifiers.
    all_cities_list = list(set(parent1))
    if len(all_cities_list) != size:
        # This case implies parent1 had duplicate cities, which is unusual for a TSP tour.
        # Or, parent1 and parent2 do not contain the same set of cities.
        # For simplicity, we proceed assuming parent1 is a valid permutation.
        # A more robust check might be `set(parent1) == set(parent2)`.
        logger.warning(
            f"ERX: Number of unique cities in parent1 ({len(all_cities_list)}) does not match tour size ({size})."
        )
        # Ensure all_cities_list has the correct items if parents differ in sets
        all_cities_set = set(parent1) | set(parent2)
        if len(all_cities_set) != size:
            logger.error(
                f"ERX: Combined unique cities from parents ({len(all_cities_set)}) does not match tour size ({size}). Cannot proceed reliably."
            )
            return list(parent1)  # Fallback
        all_cities_list = list(all_cities_set)

    # 1. Build Adjacency Map
    # adj_map_erx[city_X] = {neighbor_Y: type}
    # type = 2 if (X,Y) is an adjacency in both parents (common edge)
    # type = 1 if (X,Y) is an adjacency in only one parent

    adj_map_erx: Dict[int, Dict[int, int]] = {city: {} for city in all_cities_list}

    # Helper to get adjacencies for a single parent
    def get_adjacencies_for_parent(
        p: Sequence[int], cities_in_tour: List[int]
    ) -> Dict[int, Set[int]]:
        adj: Dict[int, Set[int]] = {city: set() for city in cities_in_tour}
        if not p:
            return adj
        current_tour_size = len(p)
        if current_tour_size == 0:
            return adj

        for i in range(current_tour_size):
            u = p[i]
            # In a tour ... -> prev_city -> u -> next_city -> ...
            # u is adjacent to prev_city and next_city
            prev_city = p[(i - 1 + current_tour_size) % current_tour_size]
            next_city = p[(i + 1 + current_tour_size) % current_tour_size]

            # Add adjacencies, sets will handle cases like size=2 where prev_city == next_city
            adj[u].add(prev_city)
            adj[u].add(next_city)
        return adj

    adj_p1 = get_adjacencies_for_parent(parent1, all_cities_list)
    adj_p2 = get_adjacencies_for_parent(parent2, all_cities_list)

    for city_c in all_cities_list:
        neighbors_in_p1 = adj_p1.get(city_c, set())
        neighbors_in_p2 = adj_p2.get(city_c, set())

        all_potential_neighbors = (
            neighbors_in_p1 | neighbors_in_p2
        )  # Union of neighbors

        for neighbor_n in all_potential_neighbors:
            is_in_p1 = neighbor_n in neighbors_in_p1
            is_in_p2 = neighbor_n in neighbors_in_p2

            if is_in_p1 and is_in_p2:
                adj_map_erx[city_c][neighbor_n] = 2  # Common adjacency
            elif is_in_p1 or is_in_p2:  # Must be in one if in the union
                adj_map_erx[city_c][neighbor_n] = 1  # Adjacency unique to one parent

    child: List[int] = []
    unvisited_cities = set(all_cities_list)

    # 2. Select starting city
    # Using parent1[0] for consistency with how some examples might start.
    # A more typical ERX might use `random.choice(all_cities_list)`.
    current_city_in_child = parent1[0]

    child.append(current_city_in_child)
    unvisited_cities.remove(current_city_in_child)

    # 3. Iteratively build the child tour
    while len(child) < size:
        # A. Remove current_city_in_child from all other cities' adjacency lists in adj_map_erx.
        # This means current_city_in_child can no longer be selected as a neighbor by any other unvisited city.
        for city_k in all_cities_list:
            if (
                city_k != current_city_in_child
                and current_city_in_child in adj_map_erx.get(city_k, {})
            ):
                del adj_map_erx[city_k][current_city_in_child]

        # B. Select the next city to add to the child tour
        # These are neighbors of the `current_city_in_child` that are still in `unvisited_cities`.
        potential_neighbors_of_current = adj_map_erx.get(current_city_in_child, {})

        candidate_neighbors: Dict[int, int] = (
            {}
        )  # Stores {neighbor_node: type_from_adj_map (1 or 2)}
        for neighbor, type_val in potential_neighbors_of_current.items():
            if neighbor in unvisited_cities:
                candidate_neighbors[neighbor] = type_val

        next_city_chosen = -1  # Sentinel to ensure a city is chosen

        if not candidate_neighbors:
            # Case: current_city_in_child has no unvisited neighbors in its list (it's isolated).
            # Pick a random unvisited city from the entire set.
            if unvisited_cities:
                next_city_chosen = random.choice(list(unvisited_cities))
            else:
                # This should not happen if len(child) < size, means something went wrong.
                logger.error(
                    f"ERX Fallback: No candidates & no unvisited cities. Child: {child}, Current: {current_city_in_child}"
                )
                return list(parent1)  # Fallback
        else:
            # Determine which list of candidates to use for tie-breaking (common or unique)
            common_options = {n: t for n, t in candidate_neighbors.items() if t == 2}

            options_for_min_length_check: Dict[int, int]
            if common_options:  # Prioritize common edges
                options_for_min_length_check = common_options
            else:  # No common edges, use unique edges
                options_for_min_length_check = candidate_neighbors

            # Find the neighbor(s) in options_for_min_length_check with the fewest onward unvisited neighbors.
            min_onward_neighbors = float("inf")
            best_next_options_list = []

            for N_option_city in options_for_min_length_check.keys():
                count_of_N_option_city_onward_neighbors = 0
                # Check adj_map_erx[N_option_city] for its unvisited neighbors
                for onward_neighbor_of_N in adj_map_erx.get(N_option_city, {}).keys():
                    if onward_neighbor_of_N in unvisited_cities:
                        count_of_N_option_city_onward_neighbors += 1

                if count_of_N_option_city_onward_neighbors < min_onward_neighbors:
                    min_onward_neighbors = count_of_N_option_city_onward_neighbors
                    best_next_options_list = [N_option_city]
                elif count_of_N_option_city_onward_neighbors == min_onward_neighbors:
                    best_next_options_list.append(N_option_city)

            if best_next_options_list:
                next_city_chosen = random.choice(best_next_options_list)
            elif candidate_neighbors:
                # Fallback: if min_length_check somehow yielded nothing from non-empty options_for_min_length_check
                # (e.g., all had inf length, which is unlikely), pick randomly from available candidates.
                logger.warning(
                    f"ERX: Tie-breaking yielded no best_options from {options_for_min_length_check}. Picking from candidates: {candidate_neighbors.keys()}."
                )
                next_city_chosen = random.choice(list(candidate_neighbors.keys()))
            elif (
                unvisited_cities
            ):  # Broader fallback if all candidates somehow disappeared
                next_city_chosen = random.choice(list(unvisited_cities))
            else:  # Should not be reached
                logger.error(
                    f"ERX Fallback: No best_options, no candidates, no unvisited. Child: {child}, Current: {current_city_in_child}"
                )
                return list(parent1)

        if next_city_chosen == -1:
            # This indicates a logic flaw if reached, as fallbacks should prevent it.
            logger.error(
                f"ERX CRITICAL FAILURE: next_city_chosen remained -1. Child: {child}, Current: {current_city_in_child}"
            )
            return list(parent1)  # Critical fallback

        child.append(next_city_chosen)
        unvisited_cities.remove(next_city_chosen)
        current_city_in_child = next_city_chosen  # Update for the next iteration

    if len(child) != size:  # Final safety check
        logger.error(
            f"ERX Crossover: Produced child of wrong length ({len(child)} vs {size}). P1:{parent1}, P2:{parent2}, Child:{child}"
        )
        return list(parent1)  # Fallback

    return child
