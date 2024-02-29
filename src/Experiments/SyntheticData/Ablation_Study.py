#!/usr/bin/env python
# coding: utf-8


import os

import pandas as pd

os.sys.path.append("/home/ubuntu/automated_consensus_clustering/automated_consensus/src/")

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from ConsensusCS.ConsensusCS import build_consensus_cs
from pathlib import Path

import os
import shutil
import numpy as np

from automlclustering.ClusterValidityIndices import CVIHandler


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


k_range = (2, 100)
n_consensus_loops = 5
n_generation_loops = n_consensus_loops
generation_cs = ClusteringCS.ClusteringCS.KMEANS_SPACE
consensus_cs = build_consensus_cs(k_range=k_range)

dataset_file_names = [file for file in os.listdir("/volume/datasets/synthetic") if ".csv" in file]
runs = 5
summary_columns = ["Method", "Best CVI score", "Best ARI", "Best NMI", "best config", "wallclock time", "run",
                   "dataset", "cvi"]


def run_each_cf_separately(gen_ensemble):
    detailed_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    for cc_function in CC_FUNCTIONS:
        consensus_cs = build_consensus_cs(cc_functions=[cc_function], k_range=k_range)
        ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                          cs_generation=generation_cs,
                                          cs_consensus=consensus_cs,
                                          seed=seed)
        ens_optimizer.ensemble = gen_ensemble
        consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_consensus_loops, k_range=k_range)
        consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                          additional_result_info,
                                                          ground_truth_clustering=y)
        consensus_result_df["Method"] = cc_function.get_name()
        clean_up_optimizer_directory(consensus_opt)
        detailed_df = pd.concat([detailed_df, consensus_result_df])
        summary_df = summary_df.append(consensus_result_df.iloc[-1][summary_columns])
    return detailed_df, summary_df


def optimize_ensemble_size(gen_ensemble):
    detailed_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    for m in range(5, 35, 5):
        consensus_cs = build_consensus_cs(k_range=k_range, default_ensemble_size=m,
                                          max_ensemble_size=None, step_size=None)
        ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                          cs_generation=generation_cs, cs_consensus=consensus_cs, seed=seed)
        ens_optimizer.ensemble = gen_ensemble

        consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_consensus_loops, k_range=k_range)
        consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                          additional_result_info,
                                                          ground_truth_clustering=y)
        consensus_result_df["Method"] = f"CF and k (m={m})"
        clean_up_optimizer_directory(consensus_opt)
        detailed_df = pd.concat([detailed_df, consensus_result_df])
        summary_df = summary_df.append(consensus_result_df.iloc[-1][summary_columns])
    return detailed_df, summary_df


def run_optimization_all_three(ensemble):
    summary_df = pd.DataFrame()
    consensus_cs = build_consensus_cs(k_range=k_range)
    ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi, cs_generation=generation_cs,
                                      cs_consensus=consensus_cs,
                                      seed=seed)
    ens_optimizer.ensemble = ensemble
    consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_consensus_loops, k_range=k_range)
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)
    consensus_result_df["Method"] = "CF Optimization"

    print(consensus_result_df)
    clean_up_optimizer_directory(consensus_opt)
    summary_df = pd.DataFrame([consensus_result_df.iloc[-1][summary_columns]])
    return consensus_result_df, summary_df


from ConsensusCS.ConsensusCS import CC_FUNCTIONS


def optimize_k_aml(k_gen, gen_ensemble):
    summary_df = pd.DataFrame()
    consensus_cs = build_consensus_cs(k_range=(k_gen, k_gen))
    ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                      cs_generation=generation_cs, cs_consensus=consensus_cs, seed=seed)
    ens_optimizer.ensemble = gen_ensemble

    consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_consensus_loops, k_range=k_range)
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)

    consensus_result_df["Method"] = f"k from AML4C"
    consensus_result_df["k gen"] = k_gen

    clean_up_optimizer_directory(consensus_opt)
    summary_df = summary_df.append(consensus_result_df.iloc[-1][summary_columns])
    return consensus_result_df, summary_df


all_summary_df = pd.DataFrame()
summary_path = Path(f"results/optimization_study/")

for run in range(runs):
    result_path = summary_path / f"run_{run}"
    generation_dir = result_path / "generation"
    generation_dir.mkdir(exist_ok=True, parents=True)
    seed = run * 10
    if not result_path.exists():
        result_path.mkdir(exist_ok=True, parents=True)

    for data_file_name in dataset_file_names:
        generation_cs = ClusteringCS.ClusteringCS.KMEANS_SPACE

        if "type=varied" in data_file_name:
            cvi = CVICollection.CALINSKI_HARABASZ
        else:
            cvi = CVICollection.DENSITY_BASED_VALIDATION

        df = pd.read_csv(f"/volume/datasets/synthetic/{data_file_name}", index_col=None, header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        true_k = len(np.unique(y))
        n = X.shape[0]

        f = X.shape[1]
        noise = float(data_file_name.replace(".csv", "").split("-")[4].split("=")[-1])
        type_ = data_file_name.split("-")[0].split("=")[-1]

        additional_result_info = {"dataset": data_file_name,
                                  "cvi": cvi.get_abbrev(),
                                  "n": n, "f": f, "true_k": true_k,
                                  "noise": noise, "type": type_,
                                  "run": run}
        print(f"Running on data: {data_file_name}")

        print("--------------------")
        print(f"Running Generation")
        ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                          cs_generation=generation_cs, cs_consensus=consensus_cs, seed=seed)

        gen_optimizer = ens_optimizer.optimize_generation(n_loops=n_generation_loops, k_range=k_range)
        generation_result_df = process_result_to_dataframe(gen_optimizer,
                                                           additional_result_info,
                                                           ground_truth_clustering=y)
        generation_result_df["Method"] = "Generation"
        clean_up_optimizer_directory(gen_optimizer)
        print(generation_result_df)

        generation_result_df.to_csv(generation_dir / data_file_name, index=False)

        all_summary_df = all_summary_df.append(generation_result_df.iloc[-1][summary_columns])

        all_summary_df.to_csv(summary_path / "summary.csv", index=False)

        opt_methods_dict = {"cf_separately": run_each_cf_separately,
                            "m_separately": optimize_ensemble_size,
                            "k_aml": optimize_k_aml,
                            "cf_optimization": run_optimization_all_three}
        # Run separate optimizations with same ensemble generation
        for method, method_function in opt_methods_dict.items():
            print("--------------------")

            print(f"Running method: {method}")
            method_path = Path(f"{result_path}/{method}")

            if not method_path.exists():
                method_path.mkdir(exist_ok=True, parents=True)

            if method == "k_aml":
                k_gen = gen_optimizer.get_incumbent()["n_clusters"]
                detailed_df, summary_df = method_function(k_gen, ens_optimizer.ensemble)
            else:
                detailed_df, summary_df = method_function(ens_optimizer.ensemble)

            detailed_df.to_csv(method_path / data_file_name, index=False)
            all_summary_df = pd.concat([all_summary_df, summary_df])
            all_summary_df.to_csv(summary_path / "summary.csv", index=False)
