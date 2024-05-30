import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib import colormaps


def display_attributes(G, pos,title):
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # attitude graph
    attitudes = [G.nodes[node]['attitude'] for node in G.nodes()]
    nx.draw(G,
            pos,
            ax=axes[0],
            with_labels=False,
            node_color=attitudes,
            cmap=colormaps['Greens'],
            edge_color='gray',
            node_size=50,
            font_size=10)
    axes[0].set_title('Agent Attitude')
    sm = plt.cm.ScalarMappable(cmap=colormaps['Greens'], norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], orientation='horizontal', pad=0.05)
    cbar.set_label('Attitude')

    # proficiency graph
    proficiencies = [G.nodes[node]['proficiency'] for node in G.nodes()]
    nx.draw(G,
            pos,
            ax=axes[1],
            with_labels=False,
            node_color=proficiencies,
            cmap=colormaps['Reds'],
            edge_color='gray',
            node_size=50,
            font_size=10)
    axes[1].set_title('Agent Proficiency')
    sm = plt.cm.ScalarMappable(cmap=colormaps['Reds'], norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation='horizontal', pad=0.05)
    cbar.set_label('Proficiency')
    plt.suptitle(title)
    plt.show()


def create_graph():
    # Params for WS small world graph
    n = 100  # Number of nodes
    k = 5  # join node to its k nearest neighbours in a ring
    p = 0.1  # prob of rewiring each edge
    G = nx.watts_strogatz_graph(n, k, p)

    # Initialize 'attitude' and 'proficiency' attributes for each node
    for node in G.nodes():
        G.nodes[node]['attitude'] = random.uniform(0, 1)
        G.nodes[node]['proficiency'] = random.uniform(0, 1)

    # Compute layout once and reuse it
    pos = nx.spring_layout(G)
    display_attributes(G, pos,"Simulation at time 0")

    # Simulating interactions between agents
    increment_value = 0.05

    num_iterations = 100

    for i in range(num_iterations):
        for node in G.nodes():
            # Sum the proficiencies and attitudes of the neighbors
            neighbor_sum = sum(
                G.nodes[neighbor]['proficiency'] + G.nodes[neighbor]['attitude'] for neighbor in G.neighbors(node))
            num_neighbors = len(list(G.neighbors(node)))

            if neighbor_sum > num_neighbors:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency'] + increment_value
            elif neighbor_sum < num_neighbors:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency'] - increment_value
            else:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency']
        # update attributes after interactions with neighbours are complete
        for node in G.nodes():
            G.nodes[node]['proficiency'] = G.nodes[node]['proficiency_next_turn']

        if i % 10 == 0:
            display_attributes(G, pos, "Simulation at time:"+str(i))


# Call the function to create and display the graphs
create_graph()
