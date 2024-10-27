from data_pp import filtered_df
from slime.dish import Dish
import matplotlib.pyplot as plt
import networkx as nx

stations = filtered_df

start_loc = (stations.at[15, "x"], stations.at[15, "y"])
dish = Dish(
    dish_shape=(max(stations.x) + 10, max(stations.y) + 10),
    foods=stations,
    start_loc=start_loc,
    mould_shape=(5, 5),
    init_mould_coverage=1,
    decay=0.2,
)

dish.animate(frames=350, interval=100, filename="350steps_pheromonevar.gif")
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
plt.savefig("slime_graph_350steps_pheromonevar.png")
