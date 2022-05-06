Data Visualization
=================================

After all data has been generated, a graph is usually created to help visualize the relationship and draw some conclusions
about the dataset. This document will show how to do this using ``seaborn`` package.

The use case described here is from the paper 
*Modelling the Joint Effect of Social Determinants and Peers on Obesity Among Canadian Adults* which is described 
in :ref:`Customizing Node Values`. The model used in the paper will be referred as obesity model.

.. Creating graphs:

Creating graphs
---------------------------------

Using ``seaborn`` to graph will require an object of class ``pandas.DataFrame``.

A csv file that has data collected from running the obesity model with different values of p, 20 replications each, is used here.
The model was also run with different types of networks, small world and scale-free, which can be indicated from
the **Type** column.
The first five rows of the file:

..  csv-table::
    :file: obesity_serial_sample.csv

The complete dataset can be found here: :download:`obesity_serial.csv<../../../obesity_serial.csv>`

The file can be loaded to a panda dataframe using ``pandas.pd.read_csv()``:

..  code-block:: python

    import pandas as pd

    runs_data = pd.read_csv("obesity_serial.csv")

The graph can be plotted and shown using both ``seaborn`` and ``matplotlib``. In this case, line plot is used. The complete
code is shown below:

..  code-block:: python

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    runs = pd.read_csv("obesity_serial.csv")
    sns.set_style("darkgrid")
    sns.lineplot(data=runs, x="p", y="Obesity", hue="Network Type", 
                err_style="bars")
    plt.show()

The result graph is displayed below:

..  image:: ../../../svg/serial_times.svg


Comparing runtimes
---------------------------------------

Graphs can be used to compare the runtime across multiple categories. 

In order to explore how different types of networks influence the runtime of the model for the parallel version,
data was collected by running the obesity model with different types of networks and increasing number of agents.
The data are then organized into a csv file. The first five rows are displayed below:

..  csv-table::
    :file: parallel_times_sample.csv

The complete csv file can be found here: :download:`obesity_parallel_gpu.csv<../../../parallel_times.csv>`

In order to use the above file for graphing, a pandas dataframe should be created and then ``seaborn`` and ``matplotlib``
to graph it. Parameter ``hue`` can be set as ``Type`` to group the data. Parameter

..  code-block:: python

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    runs = pd.read_csv("obesity_parallel_gpu.csv")
    sns.set_style("darkgrid")
    sns.lineplot(data=runs, x="Number of Agents", y="Time (minutes)", hue="Type")
    plt.show()

The resulted graph is displayed below:

..  image:: ../../../svg/parallel_times.svg
    
There are different ways to organize the dataset to help with data Visualization. More details can be found on seaborn
package page
