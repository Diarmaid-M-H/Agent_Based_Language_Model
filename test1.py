import copy
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
            edge_color='#ebebe9',
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
            edge_color='#ebebe9',
            node_size=50,
            font_size=10)
    axes[1].set_title('Agent Proficiency')
    sm = plt.cm.ScalarMappable(cmap=colormaps['Reds'], norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation='horizontal', pad=0.05)
    cbar.set_label('Proficiency')
    plt.suptitle(title)
    # Calculate average proficiency and proportion highly proficient
    average_proficiency = sum(proficiencies) / len(proficiencies)
    proportion_highly_proficient = sum(1 for p in proficiencies if p > 0.75) / len(proficiencies)

    # Add the additional graphic with the calculated values
    fig.text(0.5, 0.01,
             f"Average Proficiency: {average_proficiency:.2f} | Proportion Highly Proficient: {proportion_highly_proficient:.2%}",
             ha='center', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    # Save the combined plot as an image
    # plt.savefig(r"C:\Users\diarm\OneDrive\Desktop\school\Research\Model v0.2 - Lower proficiency taken\img" + str(time.time()) + ".png")
    plt.show()


def generateLFR(n):
    # Params for LFR benchmark graph
    tau1 = 2  # Power law exponent for the degree distribution
    tau2 = 1.1  # Power law exponent for the community size distribution
    mu = 0.1  # Fraction of intra-community edges
    min_degree = 2
    max_degree = 10

    # Generate the LFR benchmark graph
    G = nx_comm.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=20, min_community=20, seed=10)
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


def generateConnectedCaveman(num_cliques, clique_size, rewire_prob=0.0):
    """
    Generate a connected caveman graph with given number of cliques and clique size.

    Parameters:
    num_cliques (int): Number of communities (cliques).
    clique_size (int): Number of nodes per community.
    rewire_prob (float): Probability of rewiring each edge to introduce randomness.

    Returns:
    G (networkx.Graph): Generated connected caveman graph.
    """
    # Generate a connected caveman graph
    G = nx.connected_caveman_graph(num_cliques, clique_size)

    # Optionally, perturb the graph to make it less regular
    if rewire_prob > 0:
        num_edges_to_swap = int(rewire_prob * G.number_of_edges())
        G = nx.double_edge_swap(G, nswap=num_edges_to_swap, max_tries=num_edges_to_swap * 10)

    return G


def runBasicModel(G, pos, num_iterations, volatility_param):
    for i in range(num_iterations):
        for node in G.nodes():
            # Take the lower proficiency of the two.
            proficiency_sum = sum(
                min(G.nodes[neighbor]['proficiency'], G.nodes[node]['proficiency']) for neighbor in G.neighbors(node))
            attitude_sum = sum(G.nodes[neighbor]['attitude'] for neighbor in G.neighbors(node))
            neighbor_sum = proficiency_sum + attitude_sum
            num_neighbors = len(list(G.neighbors(node)))
            if neighbor_sum > num_neighbors and (G.nodes[node]['proficiency'] + (1 * volatility_param)) <= 1:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency'] + (1 * volatility_param)
            elif neighbor_sum < num_neighbors and (G.nodes[node]['proficiency'] - (1 * volatility_param)) >= 0:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency'] - (1 * volatility_param)
            else:
                G.nodes[node]['proficiency_next_turn'] = G.nodes[node]['proficiency']
        # update attributes after interactions with neighbours are complete
        for node in G.nodes():
            G.nodes[node]['proficiency'] = G.nodes[node]['proficiency_next_turn']

        # if i % 10 == 0:
        # display_attributes(G, pos, "Simulation at time:" + str(i))
    display_attributes(G, pos, "Simulation at time:" + str(num_iterations))


def connectHighAttitudeAgents(Graph, proportionToConnect, numConnections):
    # Rank nodes by attitude in descending order
    nodes_sorted_by_attitude = sorted(Graph.nodes(data=True), key=lambda x: x[1]['attitude'], reverse=True)

    # Select the top proportion of nodes
    num_nodes_to_select = int(proportionToConnect * len(nodes_sorted_by_attitude))
    high_attitude_nodes = [node[0] for node in nodes_sorted_by_attitude[:num_nodes_to_select]]

    # Add random connections for each high-attitude node
    for node in high_attitude_nodes:
        # convert to list or get typeErrors
        potential_connections = list(set(Graph.nodes()) - {node} - set(Graph.neighbors(node)))
        connections_to_add = random.sample(potential_connections, min(numConnections, len(potential_connections)))

        for target_node in connections_to_add:
            if not Graph.has_edge(node, target_node):
                Graph.add_edge(node, target_node)


def connectHighProficiencyAgents(Graph, proportionToConnect, numConnections):
    # Rank nodes by proficiency in descending order
    nodes_sorted_by_proficiency = sorted(Graph.nodes(data=True), key=lambda x: x[1]['proficiency'], reverse=True)

    # Select the top proportion of nodes
    num_nodes_to_select = int(proportionToConnect * len(nodes_sorted_by_proficiency))
    high_attitude_nodes = [node[0] for node in nodes_sorted_by_proficiency[:num_nodes_to_select]]

    # Add random connections for each high-proficiency node
    for node in high_attitude_nodes:
        # convert to list or get typeErrors
        potential_connections = list(set(Graph.nodes()) - {node} - set(Graph.neighbors(node)))
        connections_to_add = random.sample(potential_connections, min(numConnections, len(potential_connections)))

        for target_node in connections_to_add:
            if not Graph.has_edge(node, target_node):
                Graph.add_edge(node, target_node)



def create_graph():
    # general params (specific params are in the relevant generation function
    n = 200  # Number of nodes

    # WS - Watts Strogatz
    # G = generateWS(n)
    # LFR - Lancichinetti–Fortunato–Radicchi benchmark (community structure graph)
    #G = generateLFR(n)
    G = generateConnectedCaveman(20, 10, 0.01)

    # Initialize 'attitude' and 'proficiency' attributes for each node
    for node in G.nodes():
        G.nodes[node]['attitude'] = random.uniform(0, 1)
        G.nodes[node]['proficiency'] = random.uniform(0, 1)

    # Compute layout once and reuse it
    pos = nx.spring_layout(G)

    # deep copy isntead of regular reference copy.
    connectedHighProficiency = copy.deepcopy(G)
    connectedHighAttitude = copy.deepcopy(G)
    connectHighAttitudeAgents(connectedHighAttitude, 0.1, 3)
    connectHighProficiencyAgents(connectedHighProficiency, 0.1, 3)

    # Simulating interactions between agents
    volatility_param = 0.1
    num_iterations = 1000

    #run model on original graph
    runBasicModel(G, pos, num_iterations, volatility_param)

    #run model on graph with high proficiency agents connected
    runBasicModel(connectedHighProficiency, pos, num_iterations, volatility_param)

    #run model on graph with high attitude agents connected
    runBasicModel(connectedHighAttitude, pos, num_iterations, volatility_param)


# Call the function to create and display the graphs
create_graph()

# TODO: change increment_value to volatility -- DONE
# TODO: write up results for model in its current state -- DONE
# TODO: Change interaction code so the lower proficiency of the two is taken -- DONE
# TODO: add prestige variable
