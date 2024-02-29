# Get datasets
# Generate data
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS.ConsensusCS import build_consensus_cs, CC_FUNCTIONS
from Utils.Utils import get_type_from_dataset, get_noise_from_dataset, process_result_to_dataframe, \
    clean_up_optimizer_directory
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from EnsOptimizer.EnsembleSelection import SelectionStrategies
from Experiments.SyntheticData import DataGeneration

random_state = 1234


def generate_ensemble(X_test, k_range, random_state=random_state):
    ens = np.zeros((k_range[1] - k_range[0] + 1, X_test.shape[0]))
    gen_start = time.time()
    for i, k in enumerate(list(range(k_range[0], k_range[1] + 1))):
        print(f"Running k-Means with k={k}")
        km_time = time.time()
        km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=random_state)
        y_k = km.fit_predict(X_test)
        ens[i, :] = y_k
    ens = ens.transpose()
    gen_time = time.time() - gen_start
    return ens, gen_time


if __name__ == '__main__':
    datasets = DataGeneration.generate_datasets(n_values=[1000,  # 5000,
                                                          10000, 50000,
                                                          # 70000 --> Todo: Have to extract meta-features for them
                                                          ])
    X = [data[0] for _, data in datasets.items()]
    y_labels = [data[1] for _, data in datasets.items()]

    dataset_names = [d_name for d_name, _ in datasets.items()]
    dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
    dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
    dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)
    dataset_df = dataset_df.sort_values("dataset")
    training_datasets_df, test_datasets_df = train_test_split(dataset_df, stratify=dataset_df[["type", "noise"]],
                                                              train_size=0.7, random_state=1234)
    k_range = (2, 100)
    n_optimizer_loops = 100
    result_path = Path("../results/baselines/CAS")
    max_es_size = 50

    # iterate over datasets
    for test_dataset in test_datasets_df["dataset"].values:
        dataset_index = dataset_names.index(test_dataset)
        X_test = X[dataset_index]
        y_test = y_labels[dataset_index]

        if "gaussian" in test_dataset or "varied" in test_dataset:
            cvis = [CVICollection.CALINSKI_HARABASZ]
        else:
            cvis = [CVICollection.COP_SCORE,
                    # CVICollection.DAVIES_BOULDIN
                    ]
        for cvi in cvis:
            # iterate over Consensus Function
            # for cf in [MCLA, ABV]:
            data_result_path = result_path / "all_cfs"  # cf.get_name()
            data_result_path.mkdir(exist_ok=True, parents=True)

            result_file_name = data_result_path / (test_dataset + ".csv")
            if result_file_name.is_file():
                dataset_results = pd.read_csv(result_file_name)
            else:
                dataset_results = pd.DataFrame()

            print(f"Running cf={'all'} on {test_dataset}")

            consensus_cs = build_consensus_cs(k_range=k_range, cc_functions=CC_FUNCTIONS,
                                              max_ensemble_size=max_es_size)
            print("Consensus Search space:")
            print(consensus_cs)

            # Generate Ensemble (k-Means)
            ens, gen_time = generate_ensemble(X_test, k_range)

            # Optimizer
            ens_optimizer = EnsembleOptimizer(dataset=X_test,
                                              generation_cvi=cvi,
                                              cs_generation=None,
                                              selection_strategy=SelectionStrategies.cluster_and_select,
                                              cs_consensus=consensus_cs,
                                              seed=random_state,
                                              ensemble=ens)

            # ens_optimizer.ensemble = ens
            print(ens)
            print(ens_optimizer.ensemble)
            # Optimize Consensus Function/ k-value -> Beides zusammen?
            consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_optimizer_loops,
                                                             k_range=k_range)

            # store result
            additional_result_info = {"dataset": test_dataset, "n": X_test.shape[0], "f": X_test.shape[1],
                                      "type": get_type_from_dataset(test_dataset),
                                      "noise": get_noise_from_dataset(test_dataset),
                                      "cvi": cvi.get_abbrev(), "Method": "CAS",
                                      "es_strategy": SelectionStrategies.cluster_and_select.value,
                                      "cf": "all", "gen_time": gen_time}
            consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                              additional_result_info,
                                                              ground_truth_clustering=y_test)
            dataset_results = pd.concat([dataset_results, consensus_result_df])
            dataset_results.to_csv(result_file_name, index=False)
            clean_up_optimizer_directory(optimizer_instance=consensus_opt)
