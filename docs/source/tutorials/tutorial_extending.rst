Extending FCM
===================================================================================================================

FCM models can be extended in different ways to overcome its shortcomings, one of which is **Interval-Valued Fuzzy Cognitive Maps (IVFCMs)** proposed by Hajek and Prochazka [#f1]_. This tutorial introduces an example of modifying the library to fit the FCM model used. This will be done by extending upon the previous :ref:`insurgency model<Serial>`.

Since the node values and the edge weights for the IVFCMS are defined as intervals, the initial values and edge weights 
will be initiated as followed:

..  code-block:: python

    import networkx as nx
    import cuda-hybrid as ch

    def create_insurgency_fcm():

        # there are two data about each edge, lower and upper bounds
        FCM = nx.read_edgelist('insurgency_simple.txt', nodetype=str, 
                                data=([('lower', float), ('upper', float)]), create_using=nx.DiGraph())
        
        # Each node interval will use an initial random value which then be used to create an interval
        # and the results will be bounded between [0,1]
        val = np.random.random()
        FCM.nodes["EconomicDevelopment"]["val"] = [max(0, val - 0.02), min(1, val + 0.02)]
        val = np.random.random()
        FCM.nodes["Rebelliousness"]["val"] = [max(0, val - 0.03), min(1, val + 0.03)]
        val = np.random.random()
        FCM.nodes["AbilityOfInsurgentsToControlThePopulation"]["val"] = [max(0, val - 0.04), min(1, val + 0.04)]
        return FCM
    
    # Create the ABM/FCM hybrid model using the n
    G = nx.G = nx.newman_watts_strogatz_graph(2000, 2, 2)
    for agent in G.nodes():
        G.nodes[agent]["FCM"] = create_insurgency_fcm()
    hm = ch.HybridModel(G)

For this model, node values are randomly generated but for others, there might be certain rules that must be followed. For simplicity, no such rules are proposed for this model.

Since the node values are now intervals, there are a few changes that need to be made to the library. In the original :class:`HybridModel<cuda_hybrid.HybridModel>`, there are three 2D numpy arrays: :attr:`node_val`, :attr:`node_future_val`, and :attr:`FCM_adj`, which will need to be changed to a 3D array to accommodate the intervals. In :meth:`transformNetwork<cuda_hybrid.HybridModel.transformNetwork>` method, replace:

..  code-block:: python

    self.FCM_adj = nx.to_numpy_array(G.nodes[list(G.nodes())[0]]["FCM"], dtype=np.float32)

with:

..  code-block:: python

    # Get the size of the FCM adjecency matrix, which will be a 3D array, with the last dimension set as 2
    # for lower and upper bounds
    fcm_edges = len(G.nodes[list(G.nodes())[0]]["FCM"].nodes())
    self.FCM_adj = np.zeros((fcm_edges, fcm_edges, 2), dtype=np.float32)

    # Get the edge list from the graph and populate the adjacency matrix. If there is no edge between
    # two nodes, the interval will be [0. 0.]
    edge_lst = G.nodes[list(G.nodes())[0]]["FCM"].edges.data()
    for edge in edge_lst:
        fr = self.fcm_labels[edge[0]]
        to = self.fcm_labels[edge[1]]
        weight = edge[2]
        self.FCM_adj[fr][to][0] = weight["lower"]
        self.FCM_adj[fr][to][1] = weight["upper"]

and:

..  code-block:: python

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

with:

..  code-block:: python

    # store the node values and future node values that will serve as a buffer
    fcm_edges = len(G.nodes[list(G.nodes())[0]]["FCM"].nodes())
    abm_edges = len(G.nodes())
    # Added a third dimensions of 2 for the intervals
    self.node_val = np.zeros((abm_edges, fcm_edges, 2), dtype=np.float32)
    self.node_future_val = np.zeros((abm_edges, fcm_edges, 2), dtype=np.float32)

    # create a nested loop of the FCM node attribute and the value
    for i, node in enumerate(G.nodes(data="FCM")):
        for j, fcm_node in enumerate(node[1].nodes(data="val")):
            # store this value
            self.node_val[i][j][0] = fcm_node[1][0]
            self.node_val[i][j][1] = fcm_node[1][1]
            self.node_future_val[i][j][0] = fcm_node[1][0]
            self.node_future_val[i][j][1] = fcm_node[1][1]

Another place that needs to be modified is the :func:`runFCMCUDA_comm<cuda_hybrid.runFCMCUDA_comm>` function(or :func:`runFCMCUDA<cuda_hybrid.runFCMCUDA>` if run without communities or :meth:`runFCM<cuda_hybrid.HybridModel.runFCM>` if run serially). The rules for adding and multiplying intervals can be found in the paper [#f1]_. Since the total input to a node will now have bounds, the following code:

..  code-block:: python

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


can be changed to:

..  code-block:: python

    # loop through the concept nodes in the FCM
    for concept in range(FCM_adj.shape[0]):
        # Replace weightSum with net_lower and net_upper since lists can not be create on GPU
        net_lower = 0
        net_upper = 0
        # Save the current lower and upper bounds of the node to reference later
        node_lower = node_val[agent][concept][0]
        node_upper = node_val[agent][concept][1]

        # loop through the edge values
        for edge in range(FCM_adj.shape[1]):
            # multiply inter node value intervals with the edge weight interval
            # using the multiplication rule
            lower = FCM_adj[edge][concept][0] * node_val[agent][edge][0]
            upper = max(FCM_adj[edge][concept][0] * node_val[agent][edge][1],
                        FCM_adj[edge][concept][1] * node_val[agent][edge][0])
            # add the current input to the node to the net input using the
            # addition rule
            net_lower = min(lower + net_upper,
                                upper + net_lower)
            net_upper = upper + net_upper

        # Calculate the new values using the addition rule and and apply sigmoid function
        # to both bounds
        node_lower = min(node_lower + net_upper,
                            node_upper + net_lower)
        node_future_val[agent][concept][0] = 1 / (1 + math.exp(node_lower))
        node_upper = node_upper + net_upper
        node_future_val[agent][concept][0] = 1 / (1 + math.exp(node_lower))

        # the threshold check was not mentioned in the paper so that part was removed, though the parameter threshold is still kept for minimal changes

The last function that needs attention is :func:`loadNewValuesCUDA_comm<cuda_hybrid.loadNewValuesCUDA_comm>`. Since there are now two dimensions for each node value, the following line:

..  code-block:: python

    node_val[agent_idx][concept] = node_future_val[agent_idx][concept]

can be replaced with:

..  code-block:: python

    node_val[agent_idx][concept][0] = node_future_val[agent_idx][concept][0]
    node_val[agent_idx][concept][1] = node_future_val[agent_idx][concept][1]

Running the simulation with communities
------------------------------------------

The last thing that is needed for this to work is the interaction function. In order to simplify the interaction function and introduce running with communities to accommodate more agents, the interaction function will be run in serial while the FCMs will be run in parallel. There are only a few differences between this interaction function and the interaction function in the :ref:`serial insurgency model<Serial>` because of broadcasting in numpy and will be noted with comments.

..  code-block:: python

    def econ_influence(val, influencing):
        threshold = 10
        impact = 5
        avg = 0.0
        for num in influencing:
            avg += num
        lowerThresh = 1 - threshold / 100.0
        upperThresh = 1 + threshold / 100.0
        result = val
        # avg > val * upperThresh will result in a boolean array that compare every value along the axis
        # so using np.all will check if all values in the avg array pass the threshold
        if np.all(avg > val * upperThresh):
            result += val * impact / 100.0
        elif np.all(avg < val * lowerThresh):
            result -= val * impact / 100.0
        return result

    def insurgency_influence(influencedVal, influencing):
        # nothing needs to be changed in this function
        rate = 0.1
        result = influencedVal
        for num in influencing:
            result -= rate * num
        return result

    # nothing needs to be changed in this function
    def insurgency_interact(hm):
        if hm.ABM_adj.shape[0] <= 1:
            return
        # loop through each agent
        for agent in range(hm.ABM_adj.shape[0]):
            # grab the neighbors
            friends = hm.get_neighbors(agent)
            # get the numeric index for EconomicDevelopment and AbilityOfInsurgentsToControlThePopulation
            econIdx = hm.fcm_labels["EconomicDevelopment"]
            insurgeIdx = hm.fcm_labels["AbilityOfInsurgentsToControlThePopulation"]
            econList = []
            insurgeList = []
            # get the list of values for all the neighbors for the two concepts
            for friend in friends:
                econList.append(hm.node_val[friend][econIdx])
                insurgeList.append(hm.node_val[friend][insurgeIdx])
            # agents now influence each other
            hm.node_future_val[agent][econIdx] = econ_influence(
                    hm.node_val[agent][econIdx],
                    econList
            )
            hm.node_future_val[agent][econIdx] = insurgency_influence(
                    hm.node_future_val[agent][econIdx],
                    insurgeList
            )

The simulation then can be run using the :func:`run_parallel<cuda_hybrid.HybridModel.run_parallel>` and setting ``with_community`` to ``True`` along with providing a community algorithm.

..  code-block:: python

    hm.run_parallel(["Rebelliousness"], [0.05], 10, insurgency_interact, [hm], 20, 
                True, nx.algorithms.community.greedy_modularity_communities)

The results are displayed below:

.. code-block:: console

   {'Rebelliousness': array([0.40102673, 0.5344537 ], dtype=float32)}

.. rubric:: References

.. [#f1]  Hajek, P., & Prochazka, O. (2016, July). Interval-valued fuzzy cognitive maps for supporting business decisions. In 2016 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE) (pp. 531-536). IEEE.
