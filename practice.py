import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create a directed Erdos-Renyi graph with 10 nodes and a 50% probability of edges
G = nx.erdos_renyi_graph(n=10, p=0.5, directed=True)

# Remove self-loops if any
G.remove_edges_from(nx.selfloop_edges(G))

# Draw the graph
plt.figure(figsize=(8, 6))  # Optional: Set the figure size
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black')
plt.show()
