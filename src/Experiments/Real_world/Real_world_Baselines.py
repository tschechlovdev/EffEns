#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
from pathlib import Path

import numpy as np
# from sklearn.cluster import OPTICS
import pandas as pd

from Utils.Utils import get_n_from_real_world_data
from automlclustering.ClusterValidityIndices import CVIHandler
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.ClusteringCS.ClusteringCS import CONFIG_SPACE_MAPPING, KMEANS_SPACE, DBSCAN_SPACE, \
    build_partitional_config_space, build_all_algos_space
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer

# from Helper.Helper import process_result_to_dataframe, clean_up_optimizer_directory
# In[3]:

os.sys.path.append("/home/ubuntu/automated_consensus_clustering/automated_consensus/src/")


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVIHandler.CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


def compute_nmi_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVIHandler.CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


def process_result_to_dataframe(optimizer_result, additional_info, ground_truth_clustering):
    selected_cvi = additional_info["cvi"]
    # The result of the application phase an optimizer instance that holds the history of executed
    # configurations with their runtime, cvi score, and so on.
    # We can also access the predicted clustering labels of each configuration to compute ARI.
    optimizer_result_df = optimizer_result.get_runhistory_df()
    print(optimizer_result_df)
    for key, value in additional_info.items():
        # if key == "algorithms":
        #     value = "+".join(value)
        # if key == "similar dataset":
        #     value = "+".join(value)
        optimizer_result_df[key] = value

    # optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
    # optimizer_result_df["iteration"] = [i + 1 for i in range(len(optimizer_result_df))]

    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi]
    optimizer_result_df = optimizer_result_df.sort_values("iteration")
    optimizer_result_df["wallclock time"] = optimizer_result_df["runtime"].cumsum()

    optimizer_result_df['Best CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, ground_truth_clustering)
    optimizer_result_df['NMI'] = compute_nmi_values(optimizer_result_df, ground_truth_clustering)

    optimizer_result_df['Best ARI'] = optimizer_result_df.apply(
        lambda row:
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"] == row['Best CVI score'])]["ARI"].values[0],
        axis=1)
    optimizer_result_df['Best NMI'] = optimizer_result_df.apply(
        lambda row:
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"] == row['Best CVI score'])]["NMI"].values[0],
        axis=1)

    best_row = optimizer_result_df.iloc[optimizer_result_df['CVI score'].idxmin()]
    optimizer_result_df["best config"] = str(optimizer_result.get_incumbent().get_dictionary())
    optimizer_result_df["k_pred"] = [len(np.unique(best_row["labels"])) for _ in range(len(optimizer_result_df))]
    optimizer_result_df["config_ranking"] = optimizer_result_df["CVI score"].rank()

    # We do not need the labels in the CSV file
    # optimizer_result_df = optimizer_result_df.drop("labels", axis=1)
    optimizer_result_df = optimizer_result_df.drop(selected_cvi, axis=1)
    optimizer_result_df = optimizer_result_df.drop("labels", axis=1)

    return optimizer_result_df


def clean_up_optimizer_directory(optimizer_instance):
    if os.path.exists(optimizer_instance.output_dir) and os.path.isdir(optimizer_instance.output_dir):
        shutil.rmtree(optimizer_instance.output_dir)


# In[4]:


# AML4C: 0.03
# Consensus: .97
# data_file_name = "type=varied-k=50-n=1000-d=100-noise=0.csv"

k_range = (2, 100)
n_consensus_loops = 30
n_generation_loops = n_consensus_loops
n_loops = 100

dataset_file_names = [file for file in os.listdir("/volume/datasets/real_world") if ".csv" in file]

dataset_file_names = sorted(dataset_file_names, key=get_n_from_real_world_data  # , reverse=True
                            )

print(dataset_file_names)
runs = 5
summary_columns = ["Method", "Best CVI score", "Best ARI", "Best NMI", "best config", "wallclock time", "run",
                   "dataset", "cvi"]

all_summary_df = pd.DataFrame()
summary_path = Path(f"results/Baselines/")

# If result exists, we also use it!
if (summary_path / "summary.csv").is_file():
    all_summary_df = pd.read_csv(summary_path / "summary.csv")

cs_all = build_all_algos_space(k_range=k_range)
cs_part = build_partitional_config_space(k_range=k_range)
cs_mapping = {cs_name: cs_function
              for cs_name, cs_function in CONFIG_SPACE_MAPPING.items()
              if cs_name != KMEANS_SPACE
              and cs_name != DBSCAN_SPACE}

best_cvi_file = "best_cvi_real_world_data.csv"

if os.path.isfile(best_cvi_file):
    print("File exists")
    best_cvi_df = pd.read_csv(best_cvi_file)
else:
    print("File not exists")

for run in range(runs):
    for method in cs_mapping.keys():
        result_path = summary_path / f"run_{run}"

        seed = run * 10
        if not result_path.exists():
            result_path.mkdir(exist_ok=True, parents=True)

        for data_file_name in dataset_file_names:
            print("Possible CVIs:")
            print(
                best_cvi_df[best_cvi_df["dataset"] == str(data_file_name).replace("/volume/datasets/real_world/", "")][
                    "cvi"].values)
            cvi_abbrev = \
                best_cvi_df[best_cvi_df["dataset"] == str(data_file_name).replace("/volume/datasets/real_world/", "")][
                    "cvi"].values[
                    0]
            cvis = [CVICollection.get_cvi_by_abbrev(cvi_abbrev)]
            # cvis.append(CVICollection.DENSITY_BASED_VALIDATION)
            # cvis.append(CVICollection.ADJUSTED_MUTUAL)

            for cvi in cvis:
                print("---------------------")
                result_file = result_path / method / cvi.get_abbrev() / data_file_name
                if not (result_path / method / cvi.get_abbrev()).exists():
                    (result_path / method / cvi.get_abbrev()).mkdir(exist_ok=True, parents=True)

                if result_file.exists():
                    print(f"Result for {cvi.get_abbrev()} and dataset {data_file_name} already exists")
                    print("Continue")
                    continue

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

                print(f"Running Method: {method}")
                print(f"Running on data: {data_file_name}")
                print(f"Running with cvi: {cvi.get_abbrev()}")

                aml_optimizer = SMACOptimizer(dataset=X,
                                              cvi=cvi,
                                              cs=cs_mapping[method](k_range=k_range),
                                              n_loops=n_loops,
                                              wallclock_limit=360 * 60,
                                              true_labels=y,
                                              # cutoff_time=10,
                                              seed=seed
                                              )

                aml_optimizer.optimize()
                result_df = process_result_to_dataframe(aml_optimizer,
                                                        additional_result_info,
                                                        ground_truth_clustering=y)
                clean_up_optimizer_directory(aml_optimizer)

                result_df["n"] = n
                result_df["f"] = f
                result_df["true_k"] = true_k
                result_df["Method"] = method
                result_df["run"] = run

                all_summary_df = pd.concat([all_summary_df, pd.DataFrame([result_df.iloc[-1]])])

                all_summary_df.to_csv(summary_path / "summary.csv", index=False)
                result_df.to_csv(result_file, index=False)
