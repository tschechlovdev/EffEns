import time

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration
from smac.facade.smac_hpo_facade import SMAC4HPO

from ConsensusCS.ConsensusCS import execute_consensus_clustering, build_consensus_cs, CC_FUNCTIONS
from EnsOptimizer.EnsembleSelection import SelectionStrategies
from EnsOptimizer.cas import Cas
from automlclustering.ClusterValidityIndices.CVIHandler import CVI, CVICollection, CVIType
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer


class ConsensusOptimizer(SMACOptimizer):

    def __init__(self, dataset, ensemble, wallclock_limit=120 * 60,
                 cvi=None, n_loops=None, smac=SMAC4HPO,
                 cs: ConfigurationSpace = build_consensus_cs(),
                 true_labels=None,
                 output_dir=None,
                 ens_selection: SelectionStrategies = SelectionStrategies.quality,
                 seed=2,
                 limit_resources=True):
        super().__init__(dataset=dataset, cvi=cvi, cs=cs, n_loops=n_loops,
                         output_dir=output_dir,
                         smac=smac,
                         limit_resources=limit_resources,
                         true_labels=true_labels, wallclock_limit=wallclock_limit,
                         optimization_function=consensus_smac_function,  # Specify consensus optimization function
                         seed=seed)
        self.ensemble = ensemble
        self.ens_selection = ens_selection
        self.iteration = 1

        # Don't do this for our approach -> Gets some strange behaviour
        self.limit_resources = False


def consensus_smac_function(config: Configuration, optimizer_instance: ConsensusOptimizer, **kwargs):
    print("---------------------------------")
    print(f"-------------Running iteration={optimizer_instance.iteration}---------")
    X = optimizer_instance.dataset
    ensemble = optimizer_instance.ensemble
    cvi: CVI = optimizer_instance.cvi
    true_labels = optimizer_instance.true_labels

    ensemble_size = config["m"]
    k = config["k"]
    cc_function = config["cc_function"]

    # Todo: Actually dont need selection here anymore ...
    ens_start = time.time()

    if optimizer_instance.ens_selection == SelectionStrategies.quality:
        selected_ensemble = ensemble[:, 0:ensemble_size]
    elif optimizer_instance.ens_selection == SelectionStrategies.cluster_and_select:
        try:
            selected_ensemble = Cas().cluster_and_select(base_labels=ensemble, m=ensemble_size)
        except MemoryError as e:
            print("Memory error!")
            print(e)
            print("Selecting first m from ensemble!")
            selected_ensemble = ensemble[:, 0:ensemble_size]
    else:
        # Maybe happens for default ensemble size!
        selected_ensemble = ensemble

    selection_time = time.time() - ens_start

    print(f"Executing CF={cc_function}(m={ensemble_size}) with k={k}")
    cc_start = time.time()
    try:
        y_cons, err = execute_consensus_clustering(cc_name=cc_function, Y=selected_ensemble, k_out=k)
    except MemoryError as err:
        print(f"Error: {err}")
        print(f"Setting labels to one single array!")

        y_cons = np.ones(X.shape(0))
    cc_runtime = time.time() - cc_start
    print(f"Finished {cc_function}, took {cc_runtime}s")

    print(f"Executing cvi:  {cvi.get_abbrev()}")
    cvi_start = time.time()
    if cvi.cvi_type == CVIType.INTERNAL:
        score = cvi.score_cvi(X, y_cons)
    elif cvi.cvi_type == CVIType.EXTERNAL:
        score = cvi.score_cvi(data=None, true_labels=optimizer_instance.true_labels, labels=y_cons)
    else:
        raise ValueError(f"Unknown cvi type for cvi: {cvi}")
    cvi_runtime = time.time() - cvi_start
    print(f"Finished {cvi.get_abbrev()}, took {cvi_runtime}s")

    if not isinstance(y_cons, list):
        y_cons = y_cons.tolist()

    print(f"Executed CF={cc_function}(m={ensemble_size}) and k={k}, took {cc_runtime}s")
    print(f"Score for {cvi.get_abbrev()} is: {score}")

    optimizer_instance.iteration += 1
    return score, {'cc_runtime': cc_runtime,
                   'cvi_time': cvi_runtime,
                   'labels': y_cons,
                   'ens_selection_time': selection_time,
                   'error': str(err)}


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, y = make_blobs(n_samples=100)

    kmeans_results = np.asarray([KMeans(n_clusters=k).fit_predict(X) for k in range(2, 10)])
    ensemble = kmeans_results.transpose()
    print(kmeans_results)

    # ensemble = ensemble.reshape(100, 8)
    print(ensemble)
    ensemble = np.asarray(ensemble)
    cc_cs = build_consensus_cs(CC_FUNCTIONS,
                               k_range=(2, 10),
                               max_ensemble_size=10, step_size=2)

    opt = ConsensusOptimizer(dataset=X,
                             cvi=CVICollection.ADJUSTED_MUTUAL,
                             n_loops=5,
                             ensemble=ensemble,
                             true_labels=y,
                             cs=cc_cs)
    opt.optimize()
