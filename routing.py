import networkx as nx
import numpy as np
import random
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


# Step 1: Convert Adjacency List to Graph
def create_graph_from_adj_list(adj_list):
    """
    Converts an adjacency list to a NetworkX graph.
    :param adj_list: Dictionary representing adjacency list {node: [list of connected nodes]}
    :return: NetworkX Graph object
    """
    G = nx.Graph()
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            G.add_edge(
                node, neighbor, weight=np.random.randint(1, 10)
            )  # Random edge weights
    return G


# Example adjacency list input
adj_list = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1, 6],
    4: [1, 6, 7],
    5: [2, 8],
    6: [3, 4, 9],
    7: [4, 10],
    8: [5, 11],
    9: [6],
    10: [7],
    11: [8],
}

# Create the graph
graph = create_graph_from_adj_list(adj_list)


# Step 2: Define the Environment
class BusRouteEnv(gym.Env):
    def __init__(self, graph, max_routes=8, max_stops_per_route=15):
        super(BusRouteEnv, self).__init__()
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_nodes = len(self.nodes)
        self.max_routes = max_routes
        self.max_stops_per_route = max_stops_per_route

        # Action space: Choose the next node to visit
        self.action_space = spaces.Discrete(self.num_nodes)

        # Observation space: Current node and route progress
        self.observation_space = spaces.Dict(
            {
                "current_node": spaces.Discrete(self.num_nodes),
                "route_progress": spaces.Box(
                    0, max_stops_per_route, shape=(max_routes,)
                ),
            }
        )

        # Initialize state
        self.reset()

    def reset(self):
        self.routes = [[] for _ in range(self.max_routes)]  # List of routes
        self.route_index = 0  # Current route being built
        self.visited_nodes = set()  # Set of visited nodes
        self.current_node = random.choice(self.nodes)  # Start from a random node
        return {
            "current_node": self.current_node,
            "route_progress": np.zeros(self.max_routes),
        }

    def step(self, action):
        reward = 0
        done = False

        if action in self.visited_nodes:
            reward = -1  # Penalize visiting the same node
        elif len(self.routes[self.route_index]) < self.max_stops_per_route:
            # Add node to the current route
            self.routes[self.route_index].append(action)
            self.visited_nodes.add(action)
            self.current_node = action
            reward = 1  # Reward for visiting a new node
        else:
            # Move to the next route if the current route is full
            self.route_index += 1
            if self.route_index >= self.max_routes:
                done = True  # All routes are full

        # Check if all nodes are covered
        if len(self.visited_nodes) == self.num_nodes:
            reward += 10  # Bonus for covering all nodes
            done = True

        obs = {
            "current_node": self.current_node,
            "route_progress": np.array([len(r) for r in self.routes]),
        }
        return obs, reward, done, {}

    def render(self, mode="human"):
        # Visualize the routes
        colors = ["r", "g", "b", "y", "c", "m", "orange", "purple"]
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos, with_labels=True, node_color="lightgray", edge_color="gray"
        )
        for i, route in enumerate(self.routes):
            if len(route) > 1:
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=[(route[j], route[j + 1]) for j in range(len(route) - 1)],
                    edge_color=colors[i % len(colors)],
                    width=2,
                )
        plt.show()

    def save_routes_to_file(self, file_path):
        """
        Save the bus routes to a file with comma-separated node connections.
        """
        with open(file_path, "w") as file:
            file.write("Bus Routes (Nodes in Connection Order):\n")
            for i, route in enumerate(self.routes):
                if route:
                    route_line = f"Route {i+1}: {', '.join(map(str, route))}\n"
                    file.write(route_line)


# Step 3: Train the Model
env = BusRouteEnv(graph)

# Use Proximal Policy Optimization (PPO) for training
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Step 4: Evaluate the Model
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

# Render the results
env.render()

# Save the routes to a file
file_path = "/mnt/data/bus_routes.txt"
env.save_routes_to_file(file_path)
print(f"Bus routes have been saved to: {file_path}")
