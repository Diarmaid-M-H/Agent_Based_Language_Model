import copy
import time
import networkx as nx
import networkx.generators.community as nx_comm
import matplotlib.pyplot as plt
import random
from matplotlib import colormaps
import pandas as pd
import scipy as sp
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def create_custom_cmap():
    colors = ['red', 'white', 'green']  # Red for negative, white for zero, green for positive
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    return cmap


def display_attributes(G, pos, title):
    # Define colors
    majorityColour = '#ebebe9'
    neutralColour = 'green'
    minorityColour = 'red'

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(50, 25))
    # custom_cmap = create_custom_cmap()
    custom_cmap = plt.get_cmap("bwr")

    # Get edge colors based on language attribute
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if 'language' in data:
            if data['language'] == 'minority':
                edge_colors.append(minorityColour)
            elif data['language'] == 'majority':
                edge_colors.append(majorityColour)
            elif data['language'] == 'neutral':
                edge_colors.append(neutralColour)
            else:
                edge_colors.append(majorityColour)  # Default to majority color if not set correctly
        else:
            edge_colors.append(majorityColour)  # Default to majority color if attribute is missing

    # Attitude graph
    attitudes = [G.nodes[node]['attitude'] for node in G.nodes()]
    print(attitudes)
    nx.draw(G,
            pos,
            ax=axes[0],
            with_labels=False,
            node_color=attitudes,
            cmap=custom_cmap,
            edge_color=edge_colors,
            node_size=10,
            font_size=10)
    axes[0].set_title('Agent Attitude')
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-1, vmax=1, clip=False))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], orientation='horizontal', pad=0.05)
    cbar.set_label('Attitude')

    # Proficiency graph
    proficiencies = [G.nodes[node]['proficiency'] for node in G.nodes()]
    nx.draw(G,
            pos,
            ax=axes[1],
            with_labels=False,
            node_color=proficiencies,
            cmap=custom_cmap,
            edge_color=edge_colors,
            node_size=10,
            font_size=10)
    axes[1].set_title('Agent Proficiency')
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation='horizontal', pad=0.05)
    cbar.set_label('Proficiency')

    # Add legend for edge colors
    legend_elements = [
        Line2D([0], [0], color=majorityColour, lw=4, label='Majority'),
        Line2D([0], [0], color=neutralColour, lw=4, label='Neutral'),
        Line2D([0], [0], color=minorityColour, lw=4, label='Minority')
    ]
    plt.legend(handles=legend_elements, title="Agent interaction (edge) language", loc='center left',
               bbox_to_anchor=(1, 0.5))

    plt.suptitle(title)
    # Save the combined plot as an image
    plt.savefig(r"C:\Users\diarm\OneDrive\Desktop\school\Research\ConventionRun1\img" + title + ".png")
    plt.show(block=False)


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


def generateConnectedCaveman(n, clique_size, rewire_prob=0.0):
    """
    Generate a connected caveman graph with given number of nodes and clique size.

    Parameters:
    n (int): Number of nodes in the graph.
    clique_size (int): Number of nodes per community (clique).
    rewire_prob (float): Probability of rewiring each edge to introduce randomness.

    Returns:
    G (networkx.Graph): Generated connected caveman graph.
    """
    # Determine the number of cliques
    num_cliques = n // clique_size
    remainder = n % clique_size

    # Create initial connected caveman graph
    G = nx.Graph()
    start = 0
    for i in range(num_cliques):
        end = start + clique_size
        G.add_nodes_from(range(start, end))
        clique = nx.complete_graph(range(start, end))
        G.add_edges_from(clique.edges)
        start = end

    # Distribute remaining nodes across cliques
    if remainder > 0:
        extra_nodes = list(range(n - remainder, n))
        clique_index = 0
        for node in extra_nodes:
            G.add_node(node)
            # Connect the extra node to an existing clique
            G.add_edge(node, clique_index * clique_size)
            clique_index = (clique_index + 1) % num_cliques

    # Make sure the graph is connected
    if not nx.is_connected(G):
        # Connect remaining nodes to make the graph connected
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            G.add_edge(next(iter(components[i])), next(iter(components[i + 1])))

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


def addAttributes(G):
    proficiency_mean = 0.01
    proficiency_standard_deviation = 0.1

    attitude_mean = 0.1
    attitude_standard_deviation = 0.1
    # Initialize 'attitude' and 'proficiency' attributes for each node

    # random distribution
    # for node in G.nodes():
    # G.nodes[node]['attitude'] = random.uniform(0, 1)
    # G.nodes[node]['proficiency'] = random.uniform(0, 1)

    # normal distribution
    for node in G.nodes():
        attitude = random.normalvariate(mu=attitude_mean, sigma=attitude_standard_deviation)
        proficiency = random.normalvariate(mu=proficiency_mean, sigma=proficiency_standard_deviation)
        # Ensure the values are within the [0, 1] range
        attitude = min(max(attitude, 0), 1)
        proficiency = min(max(proficiency, 0), 1)

        G.nodes[node]['attitude'] = attitude
        G.nodes[node]['proficiency'] = proficiency


def neutralConvention(G, edge):
    node1, node2 = edge

    # Retrieve node attributes
    proficiency1 = G.nodes[node1]['proficiency']
    proficiency2 = G.nodes[node2]['proficiency']
    attitude1 = G.nodes[node1]['attitude']
    attitude2 = G.nodes[node2]['attitude']

    # Calculate minimum proficiency
    minProficiency = min(proficiency1, proficiency2)

    # Calculate conversation preferences
    conversationPreference1 = attitude1 + minProficiency
    conversationPreference2 = attitude2 + minProficiency

    # Determine the language based on the sum of conversation preferences
    totalPreference = conversationPreference1 + conversationPreference2
    if totalPreference > 0:
        language = 'minority'
    elif totalPreference < 0:
        language = 'majority'
    else:
        language = 'neutral'

    # Set the edge attribute
    G.edges[node1, node2]['language'] = language

    return edge


def minorityBiasedConvention(G, edge):
    node1, node2 = edge

    # Retrieve node attributes
    proficiency1 = G.nodes[node1]['proficiency']
    proficiency2 = G.nodes[node2]['proficiency']
    attitude1 = G.nodes[node1]['attitude']
    attitude2 = G.nodes[node2]['attitude']

    # Calculate minimum proficiency
    minProficiency = min(proficiency1, proficiency2)

    # Calculate conversation preferences
    conversationPreference1 = attitude1 + minProficiency
    conversationPreference2 = attitude2 + minProficiency

    if conversationPreference1 < 0 and conversationPreference2 < 0:
        language = 'majority'
    else:
        language = 'minority'

    # Set the edge attribute
    G.edges[node1, node2]['language'] = language

    return edge


def majorityBiasedConvention(G, edge):
    node1, node2 = edge

    # Retrieve node attributes
    proficiency1 = G.nodes[node1]['proficiency']
    proficiency2 = G.nodes[node2]['proficiency']
    attitude1 = G.nodes[node1]['attitude']
    attitude2 = G.nodes[node2]['attitude']

    # Calculate minimum proficiency
    minProficiency = min(proficiency1, proficiency2)

    # Calculate conversation preferences
    conversationPreference1 = attitude1 + minProficiency
    conversationPreference2 = attitude2 + minProficiency

    if conversationPreference1 > 0 and conversationPreference2 > 0:
        language = 'minority'
    else:
        language = 'majority'

    # Set the edge attribute
    G.edges[node1, node2]['language'] = language

    return edge


def createGraphFromFile(Electoral_Division="TULLAMORE RURAL", divisor=10):
    # Read the file "SAPS_2022_RAW.csv"
    try:
        df = pd.read_csv("SAPS_2022_RAW.csv", encoding='ISO-8859-1')
    except FileNotFoundError:
        print("File not found")
        return

    # Find the row where the "GEOGDESC" value matches the "Electoral_Division" parameter
    row = df[df["GEOGDESC"] == Electoral_Division]

    # Check if the row is empty
    if row.empty:
        print("electoral division not found")
        return

    # Create a new DataFrame for ED
    ED = pd.DataFrame()

    # Add the GEOGDESC value from row to ED and rename it to "name"
    ED["name"] = row["GEOGDESC"].values

    # Add the GUID value from row to ED
    ED["GUID"] = row["GUID"].values

    # Daily speakers = "very high proficiency"
    ED["very high proficiency"] = row["T3_2DIDOT"].values + row["T3_2DOEST"].values

    # Weekly speakers = "high proficiency"
    ED["high proficiency"] = row["T3_2DIWOT"].values + row["T3_2WOEST"].values

    # Less often speakers = "medium proficiency"
    ED["medium proficiency"] = row["T3_2DILOOT"].values + row["T3_2LOOEST"].values

    # Speak never or not answered = "low proficiency"
    ED["low proficiency"] = row["T3_2DIT"].values + row["T3_2NOEST"].values + row["T3_2NST"].values

    # Cannot speak Irish = "zero proficiency"
    ED["zero proficiency"] = row["T3_1NO"].values + row["T3_1NS"].values

    # Add "Total" column which contains sum of previous columns
    ED["Total"] = ED["very high proficiency"] + ED["high proficiency"] + ED["medium proficiency"] + ED[
        "low proficiency"] + ED["zero proficiency"]

    # Create a scaled version of ED
    ScaledED = ED.copy()
    for col in ["very high proficiency", "high proficiency", "medium proficiency", "low proficiency",
                "zero proficiency"]:
        ScaledED[col] = (ED[col] / divisor).round().astype(int)

    # Recalculate the "Total" column for ScaledED
    ScaledED["Total"] = ScaledED["very high proficiency"] + ScaledED["high proficiency"] + ScaledED[
        "medium proficiency"] + ScaledED["low proficiency"] + ScaledED["zero proficiency"]

    # Construct a random graph using the information from the dataframe ScaledED
    # G = generateConnectedCaveman(int(ScaledED["Total"].iloc[0]), 30)
    G = generateWS(int(ScaledED["Total"].iloc[0]))

    # Define the mapping of proficiency levels to their values
    proficiency_map = {
        "very high proficiency": 0.5,
        "high proficiency": 0.25,
        "medium proficiency": 0.00,
        "low proficiency": -0.25,
        "zero proficiency": -1
    }

    # Initialize an empty list to hold the proficiency values for all nodes
    proficiencies = []

    # Populate the proficiencies list with the appropriate number of each proficiency level
    for proficiency_level, value in proficiency_map.items():
        count = int(ScaledED[proficiency_level].iloc[0])
        print(proficiency_level + ": " + str(count))
        for _ in range(count):
            proficiencies.append(value)

    # Shuffle proficiency list to ensure random distribution
    random.shuffle(proficiencies)

    # Iterate through each node in the graph and assign proficiency, attitude, individual preference.
    for i, node in enumerate(G.nodes):
        proficiency = proficiencies[i]
        # attitude = random.uniform(-1, 1)
        attitude = random.uniform(-1,1)
        preference = proficiency + attitude

        G.nodes[node]["proficiency"] = proficiency
        G.nodes[node]["attitude"] = attitude
        G.nodes[node]["individualPreference"] = preference  # ranges from -2 to +2
        # print(preference)
    pos = nx.spring_layout(G)

    for edge in G.edges:
        majorityBiasedConvention(G, edge)
    display_attributes(G, pos, "Majority Biased convention")

    for edge in G.edges:
        minorityBiasedConvention(G, edge)
    display_attributes(G, pos, "Minority Biased convention")

    for edge in G.edges:
        neutralConvention(G, edge)
    display_attributes(G, pos, "Neutral convention")


def create_graph():
    return


# Call the function to create and display the graphs
# create_graph()
createGraphFromFile()
