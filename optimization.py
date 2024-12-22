import numpy as np
import networkx as nx
import itertools
from slime.dish import Dish
from na_data_filtering import filtered_df


def all_pairs(nodes):
    """Generates all pairs of nodes."""
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v


def node_clustering(G, u):
    """Calculate local clustering coefficient for a node."""
    neighbors = list(G[u])
    k = len(neighbors)
    if k < 2:
        return 0.0

    possible_connections = k * (k - 1) // 2
    actual_connections = sum(
        1 for v, w in itertools.combinations(neighbors, 2) if G.has_edge(v, w)
    )

    return (
        actual_connections / possible_connections if possible_connections > 0 else 0.0
    )


def clustering_coefficient(G):
    """Average of the local clustering coefficients."""
    if G.number_of_nodes() == 0:
        return 0.0

    cu = [node_clustering(G, node) for node in G.nodes()]
    return np.mean(cu)


def characteristic_path_length(G):
    """Calculate average shortest path length."""
    if G.number_of_nodes() <= 1:
        return float("inf")

    try:
        # Ensure graph is connected
        if not nx.is_connected(G):
            # Get the largest connected component
            G = G.subgraph(max(nx.connected_components(G), key=len))

        path_lengths = dict(nx.all_pairs_shortest_path_length(G))

        total_length = 0
        count = 0
        for source in path_lengths:
            for target, length in path_lengths[source].items():
                if source != target:
                    total_length += length
                    count += 1

        return total_length / count if count > 0 else float("inf")

    except Exception:
        return float("inf")


def generate_null_model(G):
    """
    Generate a null model graph that preserves degree distribution.

    Args:
        G (nx.Graph): Original graph

    Returns:
        nx.Graph: Null model graph
    """
    # Use configuration model to preserve degree sequence
    degrees = [d for n, d in G.degree()]
    null_graph = nx.configuration_model(degrees)

    # Remove parallel edges and self-loops
    null_graph = nx.Graph(null_graph)

    return null_graph


def small_world_coefficient(G):
    """
    Calculate small-world coefficient with robustness.

    Args:
        G (nx.Graph): Input graph

    Returns:
        float: Small-world coefficient
    """
    if G.number_of_nodes() < 5:
        return 0.0

    try:
        # Ensure graph is connected
        if not nx.is_connected(G):
            # Get the largest connected component
            G = G.subgraph(max(nx.connected_components(G), key=len))

        # Generate null model
        null_graph = generate_null_model(G)

        # Calculate clustering coefficients
        C_real = clustering_coefficient(G)
        C_null = clustering_coefficient(null_graph)

        # Calculate characteristic path lengths
        L_real = characteristic_path_length(G)
        L_null = characteristic_path_length(null_graph)

        # Prevent division by zero
        if C_null == 0 or L_null == 0 or L_real == 0:
            return 0.0

        # Small-world coefficient calculation
        sigma = (C_real / C_null) / (L_real / L_null)

        return (
            max(0, min(sigma, 10))
            if not np.isinf(sigma) and not np.isnan(sigma)
            else 0.0
        )

    except Exception as e:
        print(f"Small-world coefficient error: {e}")
        return 0.0


def evaluate_model(params):
    """
    Evaluate the small world metric with advanced graph generation and scoring.

    Args:
        params (dict): Dictionary of parameters for model configuration

    Returns:
        dict: Detailed evaluation results including small-world score and metrics
    """
    try:
        # Flexible parameter handling with more dynamic range
        decay = max(0.01, min(params.get("decay", 0.1), 1.0))
        pheromone_scale = max(1, params.get("pheromone_scale", 15))
        pheromone_offset = max(0, params.get("pheromone_offset", 10))

        # Apply pheromone calculation with more nuanced scaling
        filtered_df["pheromone"] = (
            filtered_df["normalized_passenger_count"] * pheromone_scale
        ) + pheromone_offset

        stations = filtered_df
        dish_shape = (
            max(stations["x"]) + 10,
            max(stations["y"]) + 10,
        )

        start_loc = (filtered_df.at[15, "x"], filtered_df.at[15, "y"])

        # Create Dish with standard parameters
        dish = Dish(
            dish_shape=dish_shape,
            foods=stations,
            start_loc=start_loc,
            mould_shape=(5, 5),
            init_mould_coverage=1,
            decay=0.2,
        )

        # Extended, adaptive evolution
        for step in range(500):
            dish.mould.evolve()

        # Get food graph with potential post-processing
        graph = dish.get_food_graph()

        # Validate and process graph
        if not nx.is_connected(graph):
            graph = graph.subgraph(max(nx.connected_components(graph), key=len))

        # Minimum viable graph size
        if graph.number_of_nodes() < 5:
            return {
                "small_world_score": 0.0,
                "metrics": {
                    "graph_nodes": graph.number_of_nodes(),
                    "graph_edges": graph.number_of_edges(),
                    "clustering_coefficient": 0.0,
                    "characteristic_path_length": float("inf"),
                },
            }

        # Detailed metrics computation
        sw_coefficient = small_world_coefficient(graph)
        cluster_coef = clustering_coefficient(graph)
        char_path_length = characteristic_path_length(graph)

        # Enhanced scoring with more emphasis on small-world metric
        total_score = (
            sw_coefficient  # Direct small-world coefficient
            + 0.2 * cluster_coef  # Additional clustering contribution
            + 0.1 * (1 / (char_path_length + 1))  # Normalized path length efficiency
        )

        return {
            "small_world_score": total_score,
            "metrics": {
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "small_world_coefficient": sw_coefficient,
                "clustering_coefficient": cluster_coef,
                "characteristic_path_length": char_path_length,
            },
        }

    except Exception as e:
        print(f"Model evaluation error: {e}")
        return {"small_world_score": 0.0, "metrics": {}}


def adaptive_optimization(
    initial_params,
    max_iterations=30,
    initial_learning_rate=0.1,
    tolerance=1e-4,
):
    """
    Adaptive optimization with decay fixed at 0.2."""
    # Fix decay at 0.2
    current_params = {k: v for k, v in initial_params.items() if k != "decay"}
    current_params["decay"] = 0.2

    best_result = evaluate_model(current_params)
    best_score = best_result["small_world_score"]
    best_params = current_params.copy()

    # Updated parameter ranges without decay
    param_ranges = {
        "pheromone_scale": (1, 100),
        "pheromone_offset": (0, 100),
    }

    # Adaptive exploration with momentum and decay
    learning_rate = initial_learning_rate
    momentum = {param: 0 for param in current_params}

    print("Optimization Starting:")
    print(f"Initial Parameters: {current_params}")
    print(f"Initial Small-World Score: {best_score}")
    print("\nEpoch Progression:")
    print(
        "{:<10} {:<50} {:<15} {:<15}".format(
            "Epoch", "Current Parameters", "Current Score", "Best Score"
        )
    )
    print("-" * 85)

    for iteration in range(max_iterations):
        # Adaptive learning rate decay
        learning_rate = initial_learning_rate * (1 / (1 + 0.1 * iteration))

        # More sophisticated exploration
        for param in current_params.keys():
            if param not in param_ranges:
                continue

            # Momentum-based exploration
            for multiplier in [0.8, 1.0, 1.2]:
                exploration_params = current_params.copy()

                # Momentum-guided perturbation
                momentum_factor = momentum.get(param, 0)
                step = learning_rate * multiplier * (1 + momentum_factor)

                new_value = max(
                    param_ranges[param][0],
                    min(param_ranges[param][1], current_params[param] + step),
                )
                exploration_params[param] = new_value

                # Evaluate and compare
                result = evaluate_model(exploration_params)
                current_score = result["small_world_score"]

                # Update momentum and best parameters
                if current_score > best_score:
                    # Significant improvement tracking
                    improvement = (current_score - best_score) / best_score

                    # Update momentum more aggressively for significant improvements
                    momentum[param] = momentum.get(param, 0) * 0.9 + (0.1 * improvement)

                    best_score = current_score
                    best_params = exploration_params.copy()
                    best_result = result
                    current_params = exploration_params.copy()

                # Print detailed progression
                print(
                    "{:<10} {:<50} {:<15.4f} {:<15.4f}".format(
                        iteration, str(exploration_params), current_score, best_score
                    )
                )

        # More nuanced convergence check
        if iteration > 10:  # Allow more exploration initially
            improvement_rate = abs(best_score - current_score) / (best_score + 1e-10)
            if improvement_rate < tolerance:
                print(f"\nConvergence reached. Improvement rate: {improvement_rate}")
                break

    # Final reporting
    print("\nOptimization Complete:")
    print("Best Parameters:", best_params)
    print("Best Small-World Score:", best_score)
    print("\nDetailed Metrics:")
    for metric, value in best_result["metrics"].items():
        print(f"  {metric}: {value}")

    return best_params, best_score, best_result


# Initial hyperparameters (decay now set externally)
initial_params = {"pheromone_scale": 15, "pheromone_offset": 10}

# Execute optimization
best_params, best_score, best_result = adaptive_optimization(initial_params)
