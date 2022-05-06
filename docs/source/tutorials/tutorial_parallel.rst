Parallel
=============================================================

The model used here is similar to the one in the serial use case, except that the
simulation will be run for 100 steps and there will be 1000 agents. This is to show the difference
between the runtimes of the two versions. 

..  note::

    For more details about the model, please refer to :ref:`the serial use case<Serial>`.

In a python file, import the module:

..  code-block:: python

    import cuda_hybrid as ch

The model can be built using :func:`generate_model<cuda_hybrid.generate_model>`:

..  code-block:: python

    hm = ch.generate_model(1000, 'insurgency_simple.txt', 'watts')

After that, create interaction function. There can be multiple helper functions, but everything should be wrapped
in one function only to passed as an argument. For this tutorial, the interaction is serial, while the FCMs will be run
in parallel. These functions are exactly the same as the ones in :ref:`Serial`

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
        if avg > val * upperThresh:
            result += val * impact / 100.0
        elif avg < val * lowerThresh:
            result -= val * impact / 100.0
        return result

    def insurgency_influence(influencedVal, influencing):
        rate = 0.1
        result = influencedVal
        for num in influencing:
            result -= rate * num
        return result

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

The ``insurgency_interact`` can then be passed to :meth:`run_parallel<cuda_hybrid.HybridModel.run_parallel>`, along with other arguments, which includes the hybridmodel object,
a list of focus nodes, a list of thresholds for the focus nodes, the maximum number of iterations for the FCMs, and, optionally, 
number of steps.

..  code-block:: python

    print(hm.run_parallel(["Rebelliousness"], [0.05], 10, insurgency_interact, [hm], 100)])

The function :func:`run_parallel<cuda_hybrid.HybridModel.run_parallel>` will return an average values across all agents
for all the focus nodes. The results of the above simulation is shown below:

.. code-block:: console

    {'Rebelliousness': 0.9723588069677352}

The recorded time of the parallel version was ``5.16 s``, 
while that of the serial version was ``2min 44s``.


