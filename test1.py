import networkx as nx
import matplotlib.pyplot as plt

# Parameters for the WS model
n = 400  # Number of nodes
k = 5   # Each node is joined with its k nearest neighbors in a ring topology
p = 0.1 # The probability of rewiring each edge
# Generate a WS small-world graph
G = nx.watts_strogatz_graph(n, k, p)

# Draw the graph
plt.figure(figsize=(14, 8))
nx.draw(G, with_labels=False, node_color='lightblue', edge_color='gray', node_size=50, font_size=10)
plt.title('Watts-Strogatz Small-World Graph')
plt.show()
