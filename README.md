# Graph Theory Project

This project implements social networks using **NetworkX** graphs and integrates a **Random Forest** machine learning model to predict data. The influential nodes within the network are determined based on four centrality factors: 
- **Degree Centrality**
- **Eigenvector Centrality**
- **Betweenness Centrality**
- **Closeness Centrality**

After calculating the centralities for each node, their overall score is computed. The data is then scaled within a range of 0-1, and a Random Forest model is trained on it for prediction purposes.

## Project Structure

The project consists of the following main components:

### 1. **Social Network Graph Construction**
Using **NetworkX**, this part of the project creates and visualizes social network graphs. It builds a graph based on input data (which could represent a real-world social network) and allows us to examine the graph's structure and properties.

### 2. **Centrality Computations**
The project calculates the following centrality measures for each node in the graph:
- **Degree Centrality**: Measures the number of connections a node has. A node with a higher degree is considered more central.
- **Eigenvector Centrality**: Assesses a node's influence based on the number of connections it has to other influential nodes.
- **Betweenness Centrality**: Quantifies the extent to which a node lies on the shortest path between other nodes, indicating its role in communication across the network.
- **Closeness Centrality**: Indicates how close a node is to all other nodes in the network by measuring the average shortest path length from the node to all others.

### 3. **Overall Score Calculation**
Each node's centrality scores are combined into a single "score" based on the four factors. This score is used to identify influential nodes in the network.

### 4. **Data Scaling**
The node scores are scaled to a range of 0 to 1 to standardize the data for the Random Forest model.

### 5. **Threshold**
Threshhold is calculated to ensure the top 10% nodes are treated as influencers.

### 6. **Random Forest Model**
The Random Forest machine learning model is trained using the scaled data. The model is used to predict the influence of nodes and classify them into different categories based on their overall score.

## Installation and Setup

To use this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DaniyalIAhmed/Graph-Theory-Project/new/main

Date: 3-12-2024

