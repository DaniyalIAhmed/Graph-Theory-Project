import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Create a Random Graph (or replace with your dataset)
G = nx.erdos_renyi_graph(n=10, p=0.5, directed=True)

# Step 2: Preprocess the Graph
G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
if not nx.is_connected(G.to_undirected()):
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len))  # Largest connected component

# Step 3: Compute Centrality Measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

# Step 4: Create a DataFrame for Analysis
centrality_data = pd.DataFrame({
    "Node": list(degree_centrality.keys()),
    "DegreeCentrality": list(degree_centrality.values()),
    "BetweennessCentrality": list(betweenness_centrality.values()),
    "ClosenessCentrality": list(closeness_centrality.values()),
    "EigenvectorCentrality": list(eigenvector_centrality.values())
})

# Sort by Degree Centrality
centrality_data.sort_values(by="DegreeCentrality", ascending=False, inplace=True)
print("Top Influencers:\n", centrality_data.head())

# Step 5: Visualize the Network
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(
    G, pos, 
    node_size=[v * 1000 for v in degree_centrality.values()],
    cmap=plt.cm.Blues, 
    node_color=list(degree_centrality.values()),
)
nx.draw_networkx_edges(G, pos, alpha=0.68)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Network with Influencers Highlighted")
plt.show()

# Step 6: Apply K-Means Clustering
features = centrality_data[["DegreeCentrality", "BetweennessCentrality", "ClosenessCentrality", "EigenvectorCentrality"]]
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose 3 clusters
clusters = kmeans.fit_predict(features)

# Add Cluster Labels to DataFrame
centrality_data["Cluster"] = clusters

# Step 7: Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    features["DegreeCentrality"], features["BetweennessCentrality"],
    c=clusters, cmap="viridis", s=100, alpha=0.7
)
plt.title("K-Means Clustering of Nodes")
plt.xlabel("Degree Centrality")
plt.ylabel("Betweenness Centrality")
plt.colorbar(label="Cluster")
plt.show()

# Print Clustered Data
print("Clustered Data:\n", centrality_data)

