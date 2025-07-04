import matplotlib.pyplot as plt
import folium
from typing import Sequence
from pathlib import Path
from datetime import datetime


def plot_fitness_over_generations(
    best_scores: list[float],
    generations: int,
    num_generations: int,
    pop_size: int,
    mutation_rate: float,
    crossover_type: str,
    elite_count: int,
    parent_selection_count: int,
    switch_to_tournament_gen: int,
    two_opt_start_gen: int,
    best_score: int,
    tournament_size: int,
    two_opt_prob: int,
):
    current_datetime = datetime.now()
    BASE_DIR = Path(__file__).parent
    info = f"Best score: {best_score}\nPopulation: {pop_size}\nMutation Rate: {mutation_rate}\nCrossover: {crossover_type}\nGenerations: {num_generations}\nElite: {elite_count}\nParents: {parent_selection_count}\nSwitch to Tournament: {switch_to_tournament_gen}\n2-opt Start Gen: {two_opt_start_gen}\nTournament Size: {tournament_size}\nTwo opt prob: {two_opt_prob}\nFull Two Opt: True"
    plt.figure(figsize=(10, 5))
    plt.plot(best_scores, generations, marker="o", linestyle="-", label="Best Fitness")
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.axhline(
        y=200_000,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Target Score (200k)",
    )
    plt.legend()
    plt.text(
        0.98,
        0.80,
        info,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.savefig(
        BASE_DIR
        / "ga_performance"
        / f"fitness_plot{best_score}-{current_datetime.strftime("%m-%d")}.png",
        dpi=300,
        bbox_inches="tight",
    )


def visualize_route(
    route_indices: Sequence[int], coords: tuple[tuple[float, float]]
) -> folium.map:
    route_to_plot_on_map = []

    for index in route_indices:
        route_to_plot_on_map.append(coords[index])

    # Determine a reasonable map center
    if route_to_plot_on_map:
        # Calculate the average latitude and longitude for the center
        avg_lat = sum(p[0] for p in route_to_plot_on_map) / len(route_to_plot_on_map)
        avg_lon = sum(p[1] for p in route_to_plot_on_map) / len(route_to_plot_on_map)
        map_center = [avg_lat, avg_lon]
        # Adjust zoom based on the spread of points (this is a simple heuristic)
        # For a more robust solution, you might calculate bounds and use map.fit_bounds()
        zoom_level = 6
    elif coords:  # If no route but coords exist, center on the first coord
        map_center = coords[0]
        zoom_level = 10
    else:  # Fallback if everything is empty (though caught by earlier checks)
        map_center = [0, 0]
        zoom_level = 2

    my_map = folium.Map(location=map_center, zoom_start=zoom_level)
    folium.PolyLine(locations=route_to_plot_on_map, color="blue").add_to(my_map)

    for point_coord in route_to_plot_on_map:
        folium.Marker(location=point_coord).add_to(my_map)
    return my_map
