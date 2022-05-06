import networkx as nx
import numba
from numba import cuda
from numba.typed import Dict
from networkx.algorithms import community
from nptyping import NDArray, Float64
import numpy as np
import math
from typing import Tuple


class HybridModel:
    """Create a ABM/FCM hybrid model for simulation based on a created the graph
    Each node of the ABM graph will have an 'FCM' attribute that points to an FCM graph.
    Each node of an FCM graph will have a 'val' attribute that has the value of that node.

    :param ABM: The ABM/FCM graph created using networkx
    :type ABM: networkx.Graph
    """

    def __init__(self, ABM: nx.Graph):

        self.ABM = ABM
        self.transformNetwork(ABM)

    def transformNetwork(self, G: nx.Graph):
        """Use the networkx.Graph provided to extract information, including ABM and FCM adjacency matrices,
        a dictionary that maps FCM lables to their indices in the FCM adjacency matrix, along with other information

        :param G: networkx graph to be transformed
        :type G: networkx.Graph
        """
        # loop though and just get the names of the attribute for labeling
        self.fcm_labels = Dict()
        for i, node in enumerate(G.nodes[list(G.nodes())[0]]["FCM"].nodes()):
            self.fcm_labels[node] = i

        # set values of ABM and FCM matrix
        self.ABM_adj = nx.to_numpy_array(G, dtype=np.bool_)
        self.FCM_adj = nx.to_numpy_array(G.nodes[list(G.nodes())[0]]["FCM"], dtype=np.float32)

        # store the node values and future node values that will serve as a buffer
        fcm_edges = len(G.nodes[list(G.nodes())[0]]["FCM"].nodes())
        abm_edges = len(G.nodes())
        self.node_val = np.zeros((abm_edges, fcm_edges), dtype=np.float32)
        self.node_future_val = np.zeros((abm_edges, fcm_edges), dtype=np.float32)

        # create a nested loop of the FCM node attribute and the value
        for i, node in enumerate(G.nodes(data="FCM")):
            for j, fcm_node in enumerate(node[1].nodes(data="val")):
                # store this value
                self.node_val[i][j] = fcm_node[1]
                self.node_future_val[i][j] = fcm_node[1]

        self.neighbors = []
        for lst in nx.to_dict_of_lists(G).items():
            self.neighbors.append(lst[1])

    def generate_communities(self, community_algo):
        """Partition the network into communities

        :param community_algo: the community algorithm to be used
        :type community_algo: func
        """        
        communities = list(list(comm) for comm in community_algo(self.ABM))
        self.community_matrix = np.zeros([len(communities), max([len(community) for community in communities])], dtype=int)
        for i in range(self.community_matrix.shape[0]):
            for j in range(self.community_matrix.shape[1]):
                
                if j >= len(communities[i]):
                    self.community_matrix[i][j] = -1
                else:
                    self.community_matrix[i][j] = communities[i][j]

    def get_neighbors(self, node: int) -> list:
        """Get neighbors for node in an adjacency matrix representation of a graph

        :param node: the node for which neighbors are fetched
        :type node: int
        :return: a list of neighbors of the requested node
        :rtype: list
        """

        return self.neighbors[node]

    def loadNewValues(self) -> None:
        """Sets FCM values to future values for each FCM in each agent infrom numba import cuda
        the simulation
        """
        for agent in range(self.node_val.shape[0]):
            # update the new concept values of the node
            for concept in range(self.node_val.shape[1]):
                self.node_val[agent][concept] = self.node_future_val[agent][concept]

    # run fuzzy cognitive map for a given agent
    def runFCM(self, focus, threshold, max_it=100):
        """Runs fuzzy cognitive map for all agents

        :param focus: list of focus nodes
        :type focus: list
        :param threshold: a list of thresholds corresponding to each focus node
        :type threshold: list
        :param max_it: the maximum number of iterations for each agent, defaults to 100
        :type max_it: int, optional
        """
        # loop through the adjacency matrix for each agent
        for agent in range(self.ABM_adj.shape[0]):
            it = 0
            while it < max_it:
                # loop through each concept in the FCM adjacency matrix
                for concept in range(self.FCM_adj.shape[0]):
                    weightSum = 0
                    # grab each edge and value of the FCM
                    for edge in range(self.FCM_adj.shape[1]):
                        # add to the total weight
                        weightSum += (
                            self.FCM_adj[edge][concept] * self.node_val[agent][edge]
                        )

                    # update the new value of the FCM node
                    # apply tanh if it goes over the range of [0, 1]
                    num = self.node_val[agent][concept] + weightSum
                    if num > 1 or num < 0:
                        num = math.tanh(num)
                    # if np.any(self.FCM_adj[:, concept] != 0):
                    #     num = math.tanh(num)
                    self.node_future_val[agent][concept] = num

                # check all focus concepts, if all have stabilized then
                # break outer while loop
                all_stable = True
                focusInd = [self.fcm_labels[f] for f in focus]

                for i, f in enumerate(focusInd):
                    if (
                        abs(self.node_future_val[agent][f] - self.node_val[agent][f])
                        > threshold[i]
                    ):
                        all_stable = False
                        break

                if all_stable:
                    break
                # load the new values
                self.loadNewValues()
                it += 1

    def run_serial(self, focus, threshold, iters, func, argv, steps=20) -> float:
        """Run the model serially on CPU

        :param focus: a list of focus nodes
        :type focus: list(Float64)
        :param threshold: a list of thresholds for each focus nodes
        :type threshold: list(Float64)
        :param iters: the maximum number of iterations when running FCM for each agent
        :type iters: int
        :param func: the function for interaction between agents
        :type func: function
        :param argv: a list of arguments for the interaction function
        :type argv: list
        :param steps: the number of steps the model will be run, defaults to 20
        :type steps: int, optional
        :return: the average values of the focus concepts of all agents of after the simulation
        :rtype: dict
        """

        for _ in range(steps):
             # call the user written interaction/influence functions
            func(*argv)
            # load new FCM values
            self.loadNewValues()
            # now stabilize FCM for every agent
            self.runFCM(focus, threshold, iters)

        result_values = {}

        # loop through the agent values and update obesity value
        for f in focus:
            for agent in range(self.node_val.shape[0]):
                result_values[f] = (
                    result_values.get(f, 0) + self.node_val[agent][self.fcm_labels[f]]
                )
            result_values[f] /= self.node_val.shape[0]

        return result_values

    def run_parallel_community(self, focus, threshold, iters, func, args, steps, community_algo):
        """Run the interaction function and stabelize each FCM in parallel for each step

        :param focus: list concept names to stabilize
        :type focus: list
        :param threshold: stabilization thresholds for corresponding focus nodes
        :type threshold: list
        :param iters: maximum amount of times to iterate
        :type iters: int
        :param func: interaction function between agents
        :type func: function
        :param args: list of arguments for interaction function
        :type args: list
        :param steps: number of steps of the simulation
        :type steps: int
        :return: average values for each focus concept
        :rtype: Dict
        """
        
        self.generate_communities(community_algo)

        TPB = (1024, 1)
        blockspergrid_x = math.ceil(self.community_matrix.shape[0] / TPB[0])
        BPG = (blockspergrid_x, 1)

        foci = []
        for f in focus:
            foci.append(self.fcm_labels[f])
        cu_focus = cuda.to_device(foci)
        cu_community_matrix = cuda.to_device(self.community_matrix)
        cu_FCM_adj = cuda.to_device(self.FCM_adj)
        cu_threshold = cuda.to_device(threshold)

        # for each simulation step
        for _ in range(steps):
            # run the interaction function
            func(*args)
            cuda.synchronize()
            self.loadNewValues()

            # stabelize the FCM
            runFCMCUDA_comm[BPG, TPB](cu_community_matrix, cu_FCM_adj, self.node_future_val, self.node_val, iters, cu_focus, cu_threshold)
            cuda.synchronize()
    
    def run_parallel(self, focus, threshold, iters, func, args, steps = 20, with_community = False, community_algo = None):
        """Run the interaction function and stabelize each FCM in parallel for each step

        :param focus: list concept names to stabilize
        :type focus: list
        :param threshold: stabilization thresholds for corresponding focus nodes
        :type threshold: list
        :param iters: maximum amount of times to iterate
        :type iters: int
        :param func: interaction function between agents
        :type func: function
        :param args: list of arguments for interaction function
        :type args: list
        :param steps: number of steps of the simulation
        :type steps: int
        :return: average values for each focus concept
        :rtype: Dict
        """
        if with_community:
            self.run_parallel_community(focus, threshold, iters, func, args, steps, community_algo)
        else:
            TPB = (1024, 1)
            blockspergrid_x = math.ceil(self.ABM_adj.shape[0] / TPB[0])
            BPG = (blockspergrid_x, 1)

            foci = []
            for f in focus:
                foci.append(self.fcm_labels[f])
            cu_focus = cuda.to_device(foci)
            cu_ABM_adj = cuda.to_device(self.ABM_adj)
            cu_FCM_adj = cuda.to_device(self.FCM_adj)
            cu_threshold = cuda.to_device(threshold)

            # for each simulation step
            for _ in range(steps):
                # run the interaction function
                func(*args)
                cuda.synchronize()
                self.loadNewValues()

                # stabelize the FCM
                runFCMCUDA[BPG, TPB](cu_ABM_adj, cu_FCM_adj, self.node_future_val, self.node_val, iters, cu_focus, cu_threshold)
                cuda.synchronize()
        results = {}
        for f in focus:
            for agent in range(self.node_val.shape[0]):
                results[f] = results.get(f, 0) + self.node_val[agent][self.fcm_labels[f]]
            results[f] /= self.node_val.shape[0]

        return results





@cuda.jit(device=True,)
def loadNewValuesCUDA(node_val, node_future_val) -> None:
    """Set FCM values to future values for each agent in the simulation

    :param node_val: node value matrix for each agent
    :type node_val: NDArray[Float64]
    :param node_future_val: future node value matrix for each agent
    :type node_future_val: NDArray[Float64]
    """

    x = cuda.grid(1)
    if x < node_val.shape[0]:
        agent_idx = int(x) 
        for concept in range(node_val.shape[1]):
            node_val[agent_idx][concept] = node_future_val[agent_idx][concept]


@cuda.jit()
def runFCMCUDA(ABM_adj, FCM_adj, node_future_val, node_val, max_it, focus, threshold):
    """Run fuzzy cognitive map for all agents

    :param ABM_adj: _description_
    :type ABM_adj: _type_
    :param FCM_adj: Adjacency matrix for each
    :type FCM_adj: NDArray[Float64]
    :param node_future_val: future node values of each agent
    :type node_future_val: NDArray[Float64]
    :param node_val: current node values of each agent
    :type node_val: NDArray[Float64]
    :param max_it: maximum iterations for stabilization
    :type max_it: int
    :param focus: list of concept indices that are being stabelized
    :type focus: list
    :param threshold: list of corresponding threshold
    :type threshold: list
    """    
 
    # grab the position of the grid of the CUDA
    x = cuda.grid(1)
    if x < ABM_adj.shape[0]:
        agent = int(x)
        it = 0

        # loop until there isn't a significant change or reach maximum amount of iterations
        while (it < max_it):
            # loop through the concept nodes in the FCM
            for concept in range(FCM_adj.shape[0]):
                weightSum = 0

                # loop through the edge values
                for edge in range(FCM_adj.shape[1]):
                        weightSum += FCM_adj[edge][concept] * node_val[agent][edge]

                # Apply tanh if out of range and buffer the new value
                num = node_val[agent][concept] + weightSum
                if num > 1 or num < 0:
                    num = math.tanh(num)
                node_future_val[agent][concept] = num
            
            # check if all focus concepts are stable
            all_stable = True
            for i in range(len(focus)):
                if abs(node_future_val[agent][focus[i]] - node_val[agent][focus[i]]) > threshold[i]:
                    all_stable = False
                    break
            #break if all stable
            if all_stable:
                break

            # load new values and increase the iteration
            loadNewValuesCUDA(node_val, node_future_val)
            it += 1

@cuda.jit(device=True,)
def loadNewValuesCUDA_comm(node_val, node_future_val, agent_idx) -> None:
    """Set FCM values to future values for each agent in the simulation

    :param node_val: node value matrix for each agent
    :type node_val: NDArray[Float64]
    :param node_future_val: future node value matrix for each agent
    :type node_future_val: NDArray[Float64]
    """

    # x = cuda.grid(1)
    # if x < node_val.shape[0]:
    #     agent_idx = int(x) 
    for concept in range(node_val.shape[1]):
        node_val[agent_idx][concept] = node_future_val[agent_idx][concept]

@cuda.jit()
def runFCMCUDA_comm(communities, FCM_adj, node_future_val, node_val, max_it, focus, threshold):
    """Run fuzzy cognitive map for all agents

    :param ABM_adj: _description_
    :type ABM_adj: _type_
    :param FCM_adj: Adjacency matrix for each
    :type FCM_adj: NDArray[Float64]
    :param node_future_val: future node values of each agent
    :type node_future_val: NDArray[Float64]
    :param node_val: current node values of each agent
    :type node_val: NDArray[Float64]
    :param max_it: maximum iterations for stabilization
    :type max_it: int
    :param focus: list of concept indices that are being stabelized
    :type focus: list
    :param threshold: list of corresponding threshold
    :type threshold: list
    """    
 
    # grab the position of the grid of the CUDA
    x = cuda.grid(1)
    if x < communities.shape[0]:
        comm = int(x)
        for agent in communities[comm]:
            if agent == -1:
                break
            it = 0
            # loop until there isn't a significant change or reach maximum amount of iterations
            while (it < max_it):
                # loop through the concept nodes in the FCM
                for concept in range(FCM_adj.shape[0]):
                    weightSum = 0

                    # loop through the edge values
                    for edge in range(FCM_adj.shape[1]):
                            weightSum += FCM_adj[edge][concept] * node_val[agent][edge]

                    # Apply tanh if out of range and buffer the new value
                    num = node_val[agent][concept] + weightSum
                    if num > 1 or num < 0:
                        num = math.tanh(num)
                    node_future_val[agent][concept] = num
                
                # check if all focus concepts are stable
                all_stable = True
                for i in range(len(focus)):
                    if abs(node_future_val[agent][focus[i]] - node_val[agent][focus[i]]) > threshold[i]:
                        all_stable = False
                        break
                #break if all stable
                if all_stable:
                    break

                # load new values and increase the iteration
                loadNewValuesCUDA_comm(node_val, node_future_val, agent)
                it += 1
                
def create_FCM_file(filename) -> nx.DiGraph:
    """Create FCM graph for an agent

    :param filename: the path to the file for FCM model
    :type filename: string
    :return: a directed graph based on the provided file
    :rtype: networkx.DiGraph
    """

    FCM = nx.read_edgelist(
        filename, nodetype=str, data=(("weight", float),), create_using=nx.DiGraph()
    )

    for node in FCM.nodes():
        FCM.nodes[node]["val"] = np.random.random()
    return FCM


def create_graph(agent_count, graph_type, filename):
    """Create ABM graph

    :param agent_count: the number of agents in the model
    :type agent_count: int
    :param graph_type: the graph type to be created (watts, sf, newman, or barabasi)
    :type graph_type: string
    :param filename: the path to a text file that has the edges for the FCM graph
    :type filename: string
    :return: a graph for the ABM/FCM hybrid model
    :rtype: networkx.Graph
    """

    G = nx.Graph()

    # networkx functions to create our graphs
    if graph_type == "watts":
        G = nx.watts_strogatz_graph(agent_count, 4, 0)

    elif graph_type == "sf":
        G = nx.scale_free_graph(agent_count).to_undirected()
        G.remove_edges_from(list(nx.selfloop_edges(G)))

    elif graph_type == "newman":
        G = nx.newman_watts_strogatz_graph(agent_count, 2, 2)

    elif graph_type == "barabasi":
        G = nx.barabasi_albert_graph(agent_count, 2)
    else:
        sys.exit(
            "Please enter one of the following graph generators: watts, sf, newman, barabasi"
        )

    # loop through attacht the FCM
    for agent in G.nodes():
        G.nodes[agent]["FCM"] = create_FCM_file(filename)
    return G


def generate_model(agent_count, filename, graph_type):
    """Create a general model

    :param agent_count: number of agents in the model
    :type agent_count: int
    :param filename: the path to a file that stores the edges for the FCM
    :type filename: string
    :param graph_type: the type of graphs, pick from watts, sf, newman, and barabasi
    :type graph_type: string
    :return: a HybridModel
    :rtype: HybridModel
    """
    G = create_graph(agent_count, graph_type, filename)
    hm = HybridModel(G)
    return hm
