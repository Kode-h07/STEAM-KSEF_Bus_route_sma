from data_pp import filtered_df
from slime.dish import Dish
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from empiricaldist import Pmf


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
    return np.nanmean(cu)


def path_lengths(G):
    length_iter = nx.shortest_path_length(G)
    for source, dist_map in length_iter:
        for dest, dist in dist_map.items():
            if source != dest:
                yield dist


def characteristic_path_length(G):
    return np.mean(list(path_lengths(G)))


filtered_df["pheromone"] = (
    filtered_df["normalized_passenger_count"] * 27.069166599410984
) + 5.269824738631482

stations = filtered_df
# decay is hyperparameter
start_loc = (stations.at[15, "x"], stations.at[15, "y"])
dish = Dish(
    dish_shape=(max(stations.x) + 10, max(stations.y) + 10),
    foods=stations,
    start_loc=start_loc,
    mould_shape=(5, 5),
    init_mould_coverage=1,
    decay=0.1975927921532078,
)

dish.animate(frames=500, interval=100, filename="500steps_optimized.gif")
plt.show()

fig = plt.figure(figsize=(6.3, 5))
nx.draw(
    dish.food_graph,
    dish.food_positions,
    node_color="C1",
    node_shape="s",
    node_size=12,
    with_labels=False,
)
plt.savefig("slime_graph_500steps_optimized.png")

print("Clustering coefficient:")
print(
    "Graph connected by slime mould: "
    + str(clustering_coefficient(dish.get_food_graph()))
)
print()

print("Path length:")
print(
    "Graph connected by slime mould: "
    + str(characteristic_path_length(dish.get_food_graph()))
)
print()

print(
    "Number of edges in the graph connected by slime mould: "
    + str(len(dish.get_food_graph().edges))
)
