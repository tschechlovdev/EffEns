from typing import Union

import numpy as np
from ConfigSpace import ConfigurationSpace
from pandas import DataFrame
from sklearn.datasets import make_blobs

from ConsensusCS import ConsensusCS
from EnsOptimizer.ConsensusOptimizer import ConsensusOptimizer
from EnsOptimizer.EnsembleSelection import SelectionStrategies
from automlclustering.ClusterValidityIndices.CVIHandler import CVI
from automlclustering.ClusteringCS.ClusteringCS import CONFIG_SPACES, CONFIG_SPACE_MAPPING
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer


class EnsembleOptimizer():
    def __init__(self, dataset,
                 cs_generation: Union[ConfigurationSpace, str],  # PARTITIONAL, ALL_ALGOS, KMEANS_SAPCE
                 cs_consensus: ConfigurationSpace,
                 generation_cvi: CVI,
                 selection_strategy: SelectionStrategies = SelectionStrategies.quality,
                 consensus_cvi: CVI = None,
                 wallclock_limit=120 * 60,
                 true_labels=None,
                 seed=2,
                 ensemble=None,
                 limit_resources=True
                 ):

        self.dataset = dataset
        self.generation_cvi = generation_cvi

        if consensus_cvi:
            self.consensus_cvi = consensus_cvi
        else:
            self.consensus_cvi = generation_cvi

        self.cs_generation = cs_generation
        self.cs_consensus = cs_consensus

        if selection_strategy in SelectionStrategies:
            self.es_selection = selection_strategy
        else:
            raise ValueError(
                f"The selection strategy {selection_strategy} is unknown! "
                f"Select one of {SelectionStrategies}")
        self.wallclock_limit = wallclock_limit
        self.ensemble = ensemble
        self.seed = seed
        self.max_ensemble_size = None
        self.generation_optimizer: SMACOptimizer = None
        self.consensus_optimizer: SMACOptimizer = None
        self.true_labels = true_labels
        self.limit_ressources = limit_resources

    def optimize_generation(self, n_loops: int, k_range: [int, int]) -> [SMACOptimizer, DataFrame]:
        if isinstance(self.cs_generation, str):
            if self.cs_generation not in CONFIG_SPACES:
                raise ValueError(f"Unknown Configuration space for the generation {self.cs_generation}!"
                                 f"Select one of {CONFIG_SPACES}!")
            self.cs_generation = CONFIG_SPACE_MAPPING[self.cs_generation](k_range=k_range)

        self.generation_optimizer = SMACOptimizer(dataset=self.dataset,
                                                  cs=self.cs_generation,
                                                  cvi=self.generation_cvi,
                                                  n_loops=n_loops,
                                                  wallclock_limit=self.wallclock_limit,
                                                  true_labels=self.true_labels,
                                                  seed=self.seed,
                                                  limit_resources=self.limit_ressources)
        self.generation_optimizer.optimize()

        # Get run history, sort by quality
        run_history = self.generation_optimizer.get_run_history()
        run_history.sort(key=lambda entry: entry.score)

        # transform clustering results to label_matrix
        cluster_results = np.asarray([entry.labels for entry in run_history])
        self.ensemble = cluster_results.transpose()

        return self.generation_optimizer

    def _pre_select_ensemble(self) -> np.array:
        if self.ensemble is None:
            raise ValueError("No ensemble yet, first call optimize_generation()")

        if self.es_selection == SelectionStrategies.quality:
            # In this case, we do not need to take the whole ensemble. That does not work for
            # cluster and select! Here we have to calculate similarities between *all* clusterings!
            # Maybe we could do this here as well?
            if self.max_ensemble_size:
                return self.ensemble[:, :self.max_ensemble_size]
            else:
                return self.ensemble
        elif self.es_selection == SelectionStrategies.cluster_and_select:
            return self.ensemble

    def optimize_consensus(self, n_loops: int, k_range: [int, int], warmstart_configs=None):
        self.ensemble = self._pre_select_ensemble()
        print(self.cs_consensus)
        self.consensus_optimizer: SMACOptimizer = ConsensusOptimizer(dataset=self.dataset,
                                                                     cs=self.cs_consensus,
                                                                     ensemble=self.ensemble,
                                                                     cvi=self.consensus_cvi,
                                                                     n_loops=n_loops,
                                                                     true_labels=self.true_labels,
                                                                     wallclock_limit=self.wallclock_limit,
                                                                     seed=self.seed,
                                                                     ens_selection=self.es_selection,
                                                                     limit_resources=self.limit_ressources)
        self.consensus_optimizer.optimize(initial_configs=warmstart_configs)
        return self.consensus_optimizer


if __name__ == "__main__":
    X, y = make_blobs(n_samples=500)

    from ConsensusCS.ConsensusCS import build_consensus_cs
    from ClusteringCS import ClusteringCS
    from ClusterValidityIndices.CVIHandler import CVICollection
    from Optimizer.OptimizerSMAC import SMACOptimizer

    k_range = (2, 100)
    generation_cs = ClusteringCS.KMEANS_SPACE
    consensus_cs = build_consensus_cs(k_range=k_range)
    ensemble_sizes = [5]
    for m in ensemble_sizes:
        selected_ens = np.random.randint(2, 100, (500, m))
        ens_optimizer = EnsembleOptimizer(dataset=X,
                                          generation_cvi=CVICollection.ADJUSTED_RAND,
                                          true_labels=y,
                                          cs_generation=generation_cs,
                                          cs_consensus=consensus_cs,
                                          seed=2)
        ens_optimizer.ensemble = selected_ens
        ens_optimizer.optimize_consensus(n_loops=50, k_range=k_range)
        print(ens_optimizer.consensus_optimizer.get_run_history())

    consensus_cs = ConsensusCS.build_consensus_cs()
    generation_cs = ClusteringCS.ClusteringCS.KMEANS_SPACE
    ens_optimizer = EnsembleOptimizer(dataset=X,
                                      generation_cvi=CVICollection.DAVIES_BOULDIN,
                                      cs_generation=generation_cs,
                                      cs_consensus=consensus_cs
                                      )
    result = ens_optimizer.optimize_generation(n_loops=5, k_range=(2, 10))

    print(ens_optimizer.ensemble)

    ens_optimizer.optimize_consensus(n_loops=10, k_range=(2, 10))
    print(ens_optimizer.consensus_optimizer.get_run_history())
