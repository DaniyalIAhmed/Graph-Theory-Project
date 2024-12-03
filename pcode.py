import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

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
print(centrality_data)

# Step 6: Feature Scaling and Preparation for Classification
scaler = MinMaxScaler()
features = centrality_data[["BetweennessCentrality", "ClosenessCentrality", "EigenvectorCentrality"]]
features_scaled = scaler.fit_transform(features)

# Add a binary target label: Top 10% by DegreeCentrality as Influencers
threshold = centrality_data["DegreeCentrality"].quantile(0.9)
labels = (centrality_data["DegreeCentrality"] > threshold).astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
print("X-Train: ", X_train, " Y-Train: ", y_train, " Threshold: ", threshold, " Labels: ", labels)

# Step 7: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Identify the Most Influential Nodes
centrality_data['Prediction'] = model.predict(features_scaled)
print("\nInfluencers Detected by the Model:")
print(centrality_data[centrality_data['Prediction'] == 1])