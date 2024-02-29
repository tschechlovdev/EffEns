import time

import numpy as np
from pympler.tracker import SummaryTracker

from Utils import RAMManager
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection, CVIType
from automlclustering.ClusteringCS import ClusteringCS


def smac_function(config, optimizer_instance, **kwargs):
    # ... some code you want to investigate ...
    X = optimizer_instance.dataset
    cvi = optimizer_instance.cvi
    true_labels = optimizer_instance.true_labels

    algorithm_name = config["algorithm"]

    t0 = time.time()

    clust_algo_instance = ClusteringCS.ALGORITHMS_MAP[algorithm_name]
    print(f"Executing Configuration: {config}")
    e = ""
    # Execute clustering algorithm
    try:
        y = clust_algo_instance.execute_config(X, config)
    except MemoryError as e:
        print(f"Error: {e}")
        print(f"Setting labels to one single array!")
        y = np.ones(X.shape(0))
    # store additional info, such as algo and cvi runtime, and the predicted clustering labels
    algo_runtime = time.time() - t0
    print(f"Executed {str(config)}, took {algo_runtime}s")

    print(f"Start scoring {cvi.get_abbrev()}")
    cvi_start = time.time()
    # Scoring cvi, true_labels are none for internal CVI. We only use them for consistency.
    # We only want the true_labels in the learning phase, where we optimize an external CVI.
    score = cvi.score_cvi(X, labels=y, true_labels=true_labels)
    print(f"Obtained CVI score for {cvi.get_abbrev()}: {score}")
    cvi_runtime = time.time() - cvi_start
    print(f"Finished {cvi.get_abbrev()}, took {cvi_runtime}s")
    add_info = {"algo_time": algo_runtime, "metric_time": cvi_runtime, "labels": y.tolist()}

    # TODO: Just return results for our experiments
    if optimizer_instance.cvi.cvi_type == CVIType.INTERNAL:
        # if we are using an internal cvi, this is the application phase and we do not want to calculate all cvis
        # additionally
        return score, add_info

    # We are in the learning phase and thus want to calculate all CVIs
    # Actually, we should do this in the LearningPhase script!

    # TODO: For Learning Phase of AutoClust/ AutoCluster we don't need other CVIs
    # int_cvis = CVICollection.internal_cvis
    # for int_cvi in int_cvis:
    #     int_cvi_score = int_cvi.score_cvi(X, labels=y)
    #     add_info[int_cvi.get_abbrev()] = int_cvi_score
    return score, add_info
