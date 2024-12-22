import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from na_data_filtering import filtered_df
import numpy as np  # Assuming filtered_df is properly imported


def degrees(G):
    """List of degrees for nodes in `G`.

    G: Graph object

    returns: list of int
    """
    return [G.degree(u) for u in G]


def savefig(filename, **options):
    """Save the current figure.

    Keyword arguments are passed along to plt.savefig

    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

    filename: string
    """
    print("Saving figure to file", filename)
    plt.savefig(filename, **options)


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def legend(**options):
    """Draws a legend only if there is at least one labeled item.

    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

    """
    underride(options, loc="best", frameon=False)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like

    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties

    https://matplotlib.org/api/axes_api.html

    In addition, you can use `legend=False` to suppress the legend.

    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def all_pairs(nodes):
    """Generates all pairs of nodes."""
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v


def node_clustering(G, u):
    neighbors = list(G.neighbors(u))
    k = len(neighbors)

    print(f"\nNode {u}:")
    print(f"Neighbors: {neighbors}")

    if k < 2:
        print("Less than 2 neighbors, returning NaN")
        return np.nan

    # Explicitly check edges between all neighbor pairs
    edges_exist = []
    triangles = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            v, w = neighbors[i], neighbors[j]
            edge_exists = G.has_edge(v, w)
            edges_exist.append(edge_exists)
            if edge_exists:
                triangles += 1

    print(f"Edges between neighbors: {edges_exist}")
    print(f"Number of triangles: {triangles}")

    # Maximum possible edges between k neighbors
    max_possible_edges = k * (k - 1) / 2

    coefficient = triangles / max_possible_edges if max_possible_edges > 0 else 0
    print(f"Clustering coefficient: {coefficient}")

    return coefficient


def clustering_coefficient(G):
    """Average of the local clustering coefficients.

    G: Graph

    returns: float
    """
    cu = [node_clustering(G, node) for node in G]
    return np.nanmean(cu)


def path_lengths(G):
    length_iter = nx.shortest_path_length(G)
    for source, dist_map in length_iter:
        for dest, dist in dist_map.items():
            if source != dest:
                yield dist


def characteristic_path_length(G):
    return np.mean(list(path_lengths(G)))


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


# Data Import: Load the connections data from the CSV file
lines_df = pd.read_csv("lines.csv", header=None)

# Graph Creation: Create a new empty graph G
G = nx.Graph()

# Iterate over each row and treat each non-empty value as a bus station connection
for index, row in lines_df.iterrows():
    # Create a list of stations, converting to string integers
    stations = [
        str(int(float(station)))
        for station in row
        if pd.notna(station) and str(station).strip()
    ]
    # Create edges between consecutive stations in the list
    for i in range(len(stations) - 1):
        G.add_edge(stations[i], stations[i + 1])

# Optional Node Positions: Fetch positions from filtered_df with consistent type conversion
node_positions = {
    str(int(float(row["Unnamed: 0"]))): (row["x"], row["y"])
    for index, row in filtered_df.iterrows()
    if pd.notna(row["Unnamed: 0"]) and pd.notna(row["x"]) and pd.notna(row["y"])
}

# Debugging: Check which nodes are in the graph but not in node_positions
missing_nodes = [node for node in G.nodes if str(node) not in node_positions]
if missing_nodes:
    print(f"Nodes missing positions: {missing_nodes}")

# Graph Plotting: Draw the graph with custom settings
plt.figure(figsize=(10, 8))
try:
    nx.draw(
        G,
        pos=node_positions,
        node_color="C1",
        node_shape="s",
        node_size=12,
        with_labels=False,
        font_size=8,
    )
    plt.savefig("originalnetwork.png")
except nx.NetworkXError as e:
    print(f"Error: {e}")
    missing_nodes = [node for node in G.nodes if str(node) not in node_positions]
    print(f"Nodes missing positions: {missing_nodes}")


graph = G
sw_coefficient = small_world_coefficient(graph)
cluster_coef = clustering_coefficient(graph)
char_path_length = characteristic_path_length(graph)

# Enhanced scoring with more emphasis on small-world metric
total_score = (
    sw_coefficient  # Direct small-world coefficient
    + 0.2 * cluster_coef  # Additional clustering contribution
    + 0.1 * (1 / (char_path_length + 1))  # Normalized path length efficiency
)

print("Clustering coefficient:")
print("Graph connected : " + str(cluster_coef))
print()

print("Path length:")
print("Graph connected : " + str(characteristic_path_length(graph)))
print()

print("Number of edges in the graph : " + str(len(graph.edges())))

print("original graph score: " + str(total_score))
