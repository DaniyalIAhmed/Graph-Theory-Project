import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

G = nx.erdos_renyi_graph(n=500, p=0.25, directed=True)

G.remove_edges_from(nx.selfloop_edges(G))
if not nx.is_weakly_connected(G):
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

centrality_data = pd.DataFrame({
    "Node": list(degree_centrality.keys()),
    "DegreeCentrality": list(degree_centrality.values()),
    "BetweennessCentrality": list(betweenness_centrality.values()),
    "ClosenessCentrality": list(closeness_centrality.values()),
    "EigenvectorCentrality": list(eigenvector_centrality.values()),
    "cc": [nx.clustering(G, node) for node in G.nodes()]
})

combined_centrality = {}
for node in G.nodes():
    combined_score = (
        degree_centrality[node] * 0.4 +
        betweenness_centrality[node] * 0.2 +
        closeness_centrality[node] * 0.2 +
        eigenvector_centrality[node] * 0.2
    )
    combined_centrality[node] = combined_score
centrality_data["score"] = centrality_data["Node"].map(combined_centrality)
threshold = centrality_data["score"].quantile(0.9)
centrality_data['Label'] = (centrality_data["score"] > threshold).astype(int)
features = centrality_data[[
    "DegreeCentrality", "BetweennessCentrality", "ClosenessCentrality", 
    "EigenvectorCentrality", "cc", "score"
]]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
labels = centrality_data["Label"]

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
pos = nx.spring_layout(G)
influencers = centrality_data[centrality_data["Label"] == 1]["Node"]

nx.draw_networkx_nodes(
    G, pos,
    node_size= 300,
    node_color=[
        "red" if node in influencers else "blue" for node in G.nodes()
    ]
)
nx.draw_networkx_edges(G, pos, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Network Graph with Influencers Highlighted")
plt.show()
