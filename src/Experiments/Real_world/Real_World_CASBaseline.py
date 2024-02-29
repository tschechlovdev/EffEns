# Get datasets
# Generate data
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS.ConsensusCS import build_consensus_cs, CC_FUNCTIONS
from Utils.Utils import get_type_from_dataset, get_noise_from_dataset, process_result_to_dataframe, \
    clean_up_optimizer_directory, get_n_from_real_world_data
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
dataset_file_names = [file for file in os.listdir("/volume/datasets/real_world") if ".csv" in file]
dataset_file_names = sorted(dataset_file_names, key=get_n_from_real_world_data  # , reverse=True
                            )
summary_path = Path(f"results/Baselines/")
result_path = summary_path / "CAS"
result_path.mkdir(parents=True, exist_ok=True)

k_range = (2, 100)
n_optimizer_loops = 100
max_es_size = 50
runs = 5

best_cvi_file = "best_cvi_real_world_data.csv"

if os.path.isfile(best_cvi_file):
    print("File exists")
    best_cvi_df = pd.read_csv(best_cvi_file)
else:
    print("File not exists")

run = 0
seed = run * 10

# iterate over datasets
for data_file_name in dataset_file_names:
    print("-----------------")
    print(f"Dataset: {data_file_name}")
    print("Possible CVIs:")
    print(
        best_cvi_df[best_cvi_df["dataset"] == str(data_file_name).replace("/volume/datasets/real_world/", "")][
            "cvi"].values)
    cvi_abbrev = \
        best_cvi_df[best_cvi_df["dataset"] == str(data_file_name).replace("/volume/datasets/real_world/", "")][
            "cvi"].values[
            0]
    cvi = CVICollection.get_cvi_by_abbrev(cvi_abbrev)

    df = pd.read_csv(f"/volume/datasets/real_world/{data_file_name}",
                     index_col=None, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    true_k = len(np.unique(y))
    n = X.shape[0]

    f = X.shape[1]

    additional_result_info = {"dataset": data_file_name,
                              "cvi": cvi.get_abbrev(),
                              "n": n, "f": f, "true_k": true_k,
                              "run": run}

    # iterate over Consensus Function
    #for cf in [MCLA, ABV]:
    data_result_path = result_path / "all_cfs" #cf.get_name()
    data_result_path.mkdir(exist_ok=True, parents=True)

    result_file_name = data_result_path / (data_file_name)
    if result_file_name.is_file():
        print(f"File: {result_file_name} already existis")
        print("CONTINUE!")
        continue
        dataset_results = pd.read_csv(result_file_name)
    else:
        dataset_results = pd.DataFrame()

    print(f"Running cf={'all'} on {data_file_name}")

    consensus_cs = build_consensus_cs(k_range=k_range, cc_functions=CC_FUNCTIONS,
                                      max_ensemble_size=max_es_size)
    print("Consensus Search space:")
    print(consensus_cs)

    # Generate Ensemble (k-Means)
    ens, gen_time = generate_ensemble(X, k_range)

    # Optimizer
    ens_optimizer = EnsembleOptimizer(dataset=X,
                                      generation_cvi=cvi,
                                      cs_generation=None,
                                      selection_strategy=SelectionStrategies.cluster_and_select,
                                      cs_consensus=consensus_cs,
                                      seed=random_state,
                                      ensemble=ens)

    #ens_optimizer.ensemble = ens
    print(ens)
    print(ens_optimizer.ensemble)
    # Optimize Consensus Function/ k-value -> Beides zusammen?
    consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_optimizer_loops,
                                                     k_range=k_range)

    # store result
    additional_result_info = {"dataset": data_file_name, "n": X.shape[0], "f": X.shape[1],
                              #"type": get_type_from_dataset(dat),
                              #"noise": get_noise_from_dataset(test_dataset),
                              "cvi": cvi.get_abbrev(), "Method": "CAS",
                              "es_strategy": SelectionStrategies.cluster_and_select.value,
                              "cf": "all",
                              "gen_time": gen_time}
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)
    dataset_results = pd.concat([dataset_results, consensus_result_df])
    dataset_results.to_csv(result_file_name, index=False)
    clean_up_optimizer_directory(optimizer_instance=consensus_opt)
