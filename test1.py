import time
import networkx as nx
import networkx.generators.community as nx_comm
import matplotlib.pyplot as plt
import random
from matplotlib import colormaps


def display_attributes(G, pos, title):
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
    # Save the combined plot as an image
    #plt.savefig(r"C:\Users\diarm\OneDrive\Desktop\school\Research\Model v0.1 photos\img" + str(time.time()) + ".png")
    plt.show()


def generateLFR(n):
    # Params for LFR benchmark graph
    tau1 = 3  # Power law exponent for the degree distribution
    tau2 = 1.5  # Power law exponent for the community size distribution
    mu = 0.05  # Fraction of intra-community edges

    # Generate the LFR benchmark graph
    G = nx_comm.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=42)
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Ensure the graph is connected by connecting disconnected nodes to largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_nodes = list(largest_cc)
        other_components = [comp for comp in nx.connected_components(G) if comp != largest_cc]

        for comp in other_components:
            node_from_largest = random.choice(largest_cc_nodes)
            node_from_comp = random.choice(list(comp))
            G.add_edge(node_from_largest, node_from_comp)

    return G


def generateWS(n):
    # Params for WS small world graph
    k = 5  # join node to its k nearest neighbours in a ring
    p = 0.1  # prob of rewiring each edge
    G = nx.watts_strogatz_graph(n, k, p)
    return G


def create_graph():
    # general params (specific params are in the relevant generation function
    n = 200  # Number of nodes

    # WS - Watts Strogatz
    # G = generateWS(n)
    # LFR - Lancichinetti–Fortunato–Radicchi benchmark (community structure graph)
    G = generateLFR(n)

    # Initialize 'attitude' and 'proficiency' attributes for each node
    for node in G.nodes():
        G.nodes[node]['attitude'] = random.uniform(0, 1)
        G.nodes[node]['proficiency'] = random.uniform(0, 1)

    # Compute layout once and reuse it
    pos = nx.spring_layout(G)
    display_attributes(G, pos, "Simulation at time 0")

    # Simulating interactions between agents
    volatility_param = 0.1

    num_iterations = 200

    for i in range(num_iterations):
        for node in G.nodes():
            # Sum the proficiencies and attitudes of the neighbors
            neighbor_sum = sum(
                G.nodes[neighbor]['proficiency'] + G.nodes[neighbor]['attitude'] for neighbor in G.neighbors(node))
            num_neighbors = len(list(G.neighbors(node)))

            if neighbor_sum > num_neighbors:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency'] + (1 * volatility_param)
            elif neighbor_sum < num_neighbors:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency'] - (1 * volatility_param)
            else:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency']
        # update attributes after interactions with neighbours are complete
        for node in G.nodes():
            G.nodes[node]['proficiency'] = G.nodes[node]['proficiency_next_turn']

        if i % 20 == 0:
            display_attributes(G, pos, "Simulation at time:" + str(i))


# Call the function to create and display the graphs
create_graph()

# TODO: change increment_value to volatility -- DONE
# TODO: write up results for model in its current state -- DONE
# TODO: Change interaction code so the lower proficiency of the two is taken
# TODO: add prestige variable
