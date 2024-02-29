import os
import shutil
from pathlib import Path

# from sklearn.cluster import OPTICS
import pandas as pd
from sklearn.model_selection import train_test_split

from automlclustering.ClusterValidityIndices import CVIHandler
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.ClusteringCS.ClusteringCS import CONFIG_SPACE_MAPPING, KMEANS_SPACE, DBSCAN_SPACE, \
    build_all_algos_space, build_partitional_config_space
from ConsensusCS.ConsensusCS import build_consensus_cs
from Utils.Utils import get_type_from_dataset, get_noise_from_dataset
from Experiments.SyntheticData import DataGeneration
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
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


if __name__ == '__main__':
    import numpy as np

    datasets = DataGeneration.generate_datasets(n_values=[1000, 10000, 50000])

    data_X = [data[0] for _, data in datasets.items()]
    y_labels = [data[1] for _, data in datasets.items()]

    dataset_names = [d_name for d_name, _ in datasets.items()]
    dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
    dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
    dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)

    ### Train-test split!
    training_datasets_df, test_datasets_df = train_test_split(dataset_df, stratify=dataset_df[["type", "noise"]],
                                                              train_size=0.7, random_state=1234)

    print(test_datasets_df)
    print(len(test_datasets_df))

    ## Parameters
    test_data_indices = list(test_datasets_df.index)
    print(test_data_indices)

    k_range = (2, 100)
    n_consensus_loops = 40
    n_generation_loops = n_consensus_loops

    generation_cs = KMEANS_SPACE
    consensus_cs = build_consensus_cs(k_range=k_range)
    cs_all = build_all_algos_space(k_range=k_range)
    cs_part = build_partitional_config_space(k_range=k_range)
    cs_mapping = {cs_name: cs_function
                  for cs_name, cs_function in CONFIG_SPACE_MAPPING.items()
                  if cs_name != KMEANS_SPACE
                  and cs_name != DBSCAN_SPACE  # Problems with memory for n=50k and 'large' epsilon ...
                  }

    runs = 5
    for test_index in test_data_indices:
        for method in cs_mapping.keys():
            for run in range(runs):
                seed = 2 * run
                np.random.seed(seed)

                dataset_name = dataset_names[test_index]
                X = data_X[test_index]
                y_test = y_labels[test_index]
                print("----------------")
                print(f"Running method: {method}")
                if "type=varied" in dataset_name or "gaussian" in dataset_name:
                    cvis = [CVICollection.CALINSKI_HARABASZ]
                else:
                    cvis = [  # CVICollection.DAVIES_BOULDIN,
                        CVICollection.COP_SCORE
                    ]

                for cvi in cvis:

                    print("----------------")
                    print(f"Running on dataset: {dataset_name}")
                    result_path = Path(f"../results/baselines/{method}/run_{run}/{cvi.get_abbrev()}")
                    if not result_path.exists():
                        result_path.mkdir(exist_ok=True, parents=True)

                    result_file = result_path / (dataset_name + ".csv")
                    if result_file.exists():
                        print(f"Result for {result_file} already existing!")
                        print("Continue with next file")
                        continue

                    true_k = len(np.unique(y_test))
                    n = X.shape[0]

                    f = X.shape[1]
                    noise = float(dataset_name.replace(".csv", "").split("-")[4].split("=")[-1])
                    type_ = dataset_name.split("-")[0].split("=")[-1]

                    additional_result_info = {"dataset": dataset_name,
                                              "cvi": cvi.get_abbrev(),
                                              "n": n, "f": f, "true_k": true_k,
                                              "noise": noise, "type": type_,
                                              "run": run
                                              }
                    print(f"Running on data: {dataset_name}")

                    noise = float(dataset_name.replace(".csv", "").split("-")[4].split("=")[-1])

                    aml_optimizer = SMACOptimizer(dataset=X, cvi=cvi,
                                                  cs=cs_mapping[method](k_range=k_range),
                                                  n_loops=n_generation_loops + n_consensus_loops,
                                                  wallclock_limit=360 * 60,
                                                  # cutoff_time=10,
                                                  seed=seed
                                                  )

                    aml_optimizer.optimize()
                    result_df = process_result_to_dataframe(aml_optimizer,
                                                            additional_result_info,
                                                            ground_truth_clustering=y_test)
                    clean_up_optimizer_directory(aml_optimizer)
                    result_df["n"] = n
                    result_df["f"] = f
                    result_df["true_k"] = true_k
                    result_df["Method"] = method
                    result_df.to_csv(result_file, index=False)
