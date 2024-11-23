import numpy as np
from slime.dish import Dish
from data_pp import filtered_df
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import networkx as nx


def all_pairs(nodes):
    """Generates all pairs of nodes."""
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v


def node_clustering(G, u):
    neighbors = G[u]
    k = len(neighbors)
    if k < 2:
        return np.nan

    edges = [G.has_edge(v, w) for v, w in all_pairs(neighbors)]
    return np.mean(edges)


def clustering_coefficient(G):
    """Average of the local clustering coefficients.

    G: Graph

    returns: float
    """
    cu = [node_clustering(G, node) for node in G]
    # Check if the array is empty or contains NaN values
    cu = [x for x in cu if not np.isnan(x)]  # Remove NaN values
    if not cu:
        return np.nan  # Return NaN if no valid values
    return np.mean(cu)


def path_lengths(G):
    length_iter = nx.shortest_path_length(G)
    for source, dist_map in length_iter:
        for dest, dist in dist_map.items():
            if source != dest:
                yield dist


def characteristic_path_length(G):
    # Ensure there are valid paths before calculating the average
    path_lengths_list = list(path_lengths(G))
    if not path_lengths_list:
        return np.nan
    return np.mean(path_lengths_list)


# Define the evaluation function
def evaluate_model(decay, pheromone_scale, pheromone_offset):
    """
    Evaluate the slime mould simulation based on given hyperparameters.
    """
    # Adjust pheromone values in the dataset
    filtered_df["pheromone"] = (
        filtered_df["normalized_passenger_count"] * pheromone_scale
    ) + pheromone_offset

    stations = filtered_df
    dish_shape = (
        max(stations.x) + 10,
        max(stations.y) + 10,
    )  # Simplified dish shape (tuple)

    # Ensure start_loc is valid and does not cause indexing errors
    start_loc = (filtered_df.at[15, "x"], filtered_df.at[15, "y"])

    # Initialize the dish
    dish = Dish(
        dish_shape=dish_shape,
        foods=stations,
        start_loc=start_loc,
        mould_shape=(5, 5),
        init_mould_coverage=1,
        decay=decay,
    )

    # Run the slime mould simulation (no animation)
    # We assume the 'evolve' method is responsible for the simulation steps
    for _ in range(500):  # Simulate for 500 steps
        dish.mould.evolve()

    # Evaluate metrics
    graph = dish.get_food_graph()

    # Check if the graph is valid
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        print("Error: Graph has no nodes or edges!")
        return np.nan

    clustering = clustering_coefficient(graph)
    path_length = characteristic_path_length(graph)

    # Handle cases where path length might be NaN
    if np.isnan(path_length) or np.isnan(clustering):
        return np.nan

    # Small-world metric: clustering coefficient / path length
    small_world_metric = clustering / path_length if path_length > 0 else 0

    # Print evaluation details
    print(
        f"Pheromone scale: {pheromone_scale:.4f}, "
        f"Pheromone offset: {pheromone_offset:.4f}, "
        f"Clustering Coefficient: {clustering:.4f}, "
        f"Characteristic Path Length: {path_length:.4f}, "
        f"Small-World Metric: {small_world_metric:.4f}"
    )

    # Calculate score (customize this formula as needed)
    score = small_world_metric
    return -score  # Negative because we minimize in Bayesian Optimization


# Define search space for Bayesian Optimization
search_space = [
    Real(0.05, 0.2, name="decay"),  # Range for decay
    Real(10, 30, name="pheromone_scale"),  # Range for pheromone_scale
    Real(5, 15, name="pheromone_offset"),  # Range for pheromone_offset
]


# Wrapping evaluate_model for Bayesian Optimization
@use_named_args(search_space)
def objective_function(**params):
    return evaluate_model(
        params["decay"],
        params["pheromone_scale"],
        params["pheromone_offset"],
    )


# Perform Bayesian Optimization
result = gp_minimize(
    objective_function,
    dimensions=search_space,
    n_calls=50,  # Number of evaluations
    random_state=42,
)

# Best parameters and score
best_params = {
    "decay": result.x[0],
    "pheromone_scale": result.x[1],
    "pheromone_offset": result.x[2],
}
best_score = -result.fun

print("Best Hyperparameters:")
print(best_params)
print(f"Best Score: {best_score}")
