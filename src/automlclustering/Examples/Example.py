from ConfigSpace.configuration_space import Configuration
from sklearn.datasets import make_blobs

from ClusteringCS import ClusteringCS
from ClusterValidityIndices.CVIHandler import CVICollection
from Optimizer.OptimizerSMAC import SMACOptimizer

# Create a testing data set for all examples
X, y = make_blobs(n_samples=1000, n_features=10)

##########################################################
### Example 1: Running optimizer with default settings ###
# We use Hyperband in our examples
optimizer = SMACOptimizer

cs = ClusteringCS.ALL_ALGOS_SPACE

# Simple example running optimizer with default settings
automl_four_clust_instance = optimizer(dataset=X, cs=cs, n_loops=10, cvi=CVICollection.ADJUSTED_RAND, true_labels=y)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_incumbent()

##################################################################
### Example 2: Running optimizer with custom budget and metric ###
# Running optimizer with custom budget and metric
# here we use a different internal metric than the default one.
automl_four_clust_instance = optimizer(dataset=X, cvi=CVICollection.DAVIES_BOULDIN, n_loops=20)
automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_incumbent()

# It is also possible to get the history of configurations
history = automl_four_clust_instance.get_run_history()

######################################################
### Example 3: Running optimizer with warmstarting ###

# We pass the warmstart configurations as list of Configuration objects with the hyperparameter "k" and the name of the algorithm
# note that we need a configspace object for the configuration!
cs = automl_four_clust_instance.cs
k_2_config = Configuration(configuration_space=cs, values={"n_clusters": 2, "algorithm": ClusteringCS.KMEANS_ALGORITHM})
k_100_config = Configuration(configuration_space=cs, values={"n_clusters": 100, "algorithm": ClusteringCS.KMEANS_ALGORITHM})
warmstart_configs = [k_2_config, k_100_config]

# Run optimizer with the warmstart configurations
automl_four_clust_instance = optimizer(dataset=X)
automl_four_clust_instance.optimize(initial_configs=warmstart_configs)
best_configuration = automl_four_clust_instance.get_incumbent()
