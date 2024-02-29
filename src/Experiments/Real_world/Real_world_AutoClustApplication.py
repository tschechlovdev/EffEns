import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.ClusteringCS.ClusteringCS import build_config_space
from Utils.Utils import get_n_from_dataset, process_result_to_dataframe, clean_up_optimizer_directory, \
    get_n_from_real_world_data
from Experiments.SyntheticData import DataGeneration
from automlclustering.MetaLearning import MetaFeatureExtractor
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer
from automlclustering.Helper import Helper

summary_path = Path(f"results/Baselines/")
dataset_file_names = [file for file in os.listdir("/volume/datasets/real_world") if ".csv" in file]

dataset_file_names = sorted(dataset_file_names, key=get_n_from_real_world_data  # , reverse=True
                            )
n_loops = 100
k_range = (2, 100)

eval_configs = pd.read_csv("evaluated_configs.csv")
best_eval_configs = eval_configs.loc[eval_configs.groupby(["dataset"])["AMI"].idxmin()]
best_eval_configs["algorithm"] = best_eval_configs.apply(
    lambda x: ast.literal_eval(x["config"])["algorithm"],
    axis="columns")

print(best_eval_configs[["dataset", "algorithm"]])

method = f"AS -> HPO"
method = "AutoClust"
runs = 3

if method == "AutoClust":
    mf_set = "autoclust"
elif method == "AS -> HPO":
    mf_set = ["statistical", "general"]

if mf_set == "autoclust":
    meta_features = pd.read_csv("autoclust_metafeatures.csv")
elif mf_set == ["statistical", "general"]:
    meta_features = pd.read_csv("statistical_general_meta_features.csv")

print(meta_features)
training_mfs = meta_features
print(training_mfs)


def get_similar_datasets(test_mfs_values, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(training_mfs.iloc[:, :-1].to_numpy())
    distances, indices = nbrs.kneighbors(test_mfs_values.reshape(1, -1))
    print(indices)
    return training_mfs.iloc[indices[0]]["dataset"]


best_cvi_file = "best_cvi_real_world_data.csv"
if os.path.isfile(best_cvi_file):
    print("File exists")
    best_cvi_df = pd.read_csv(best_cvi_file)
else:
    print("File not exists")

for run in list(range(runs)):
    seed = 2 * run
    np.random.seed(seed)
    result_path = summary_path / f"run_{run}" / f"{method}"

    for data_file_name in dataset_file_names:
        print("Possible CVIs:")
        print(
            best_cvi_df[best_cvi_df["dataset"] == str(data_file_name).replace("/volume/datasets/real_world/", "")][
                "cvi"].values)
        cvi_abbrev = \
            best_cvi_df[best_cvi_df["dataset"] == str(data_file_name).replace("/volume/datasets/real_world/", "")][
                "cvi"].values[
                0]
        print("-------------")
        print(f"Running on dataset {data_file_name}")
        cvi = CVICollection.get_cvi_by_abbrev(cvi_abbrev)

        print("----------------")
        print(f"Running on dataset: {data_file_name}")
        result_path = Path(f"results/Baselines/{method}/run_{run}/{cvi.get_abbrev()}")
        if not result_path.exists():
            result_path.mkdir(exist_ok=True, parents=True)

        result_file = result_path / (data_file_name)
        if result_file.exists():
            print(f"Result for {result_file} already existing!")
            print("Continue with next file")
            continue
        df = pd.read_csv(f"/volume/datasets/real_world/{data_file_name}",
                         index_col=None, header=None)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1]

        mfs, test_mfs_values = MetaFeatureExtractor.extract_meta_features(X, mf_set)
        test_mfs_values = np.nan_to_num(test_mfs_values, nan=0)

        print(f"Similar Datasets:")
        similar_datasets = get_similar_datasets(test_mfs_values, k=10)
        best_conf_similar_datasets = best_eval_configs[best_eval_configs["dataset"].isin(similar_datasets)]
        print(best_conf_similar_datasets[["dataset", "algorithm", "AMI"]])
        print(best_conf_similar_datasets["algorithm"].value_counts().idxmax())
        print(best_conf_similar_datasets["algorithm"].mode())

        best_algorithm = best_conf_similar_datasets["algorithm"].value_counts().idxmax()

        cs = build_config_space(clustering_algorithms=[best_algorithm], k_range=(2, 100))

        true_k = len(np.unique(y))
        n = X.shape[0]

        f = X.shape[1]
        type_ = data_file_name.split("-")[0].split("=")[-1]

        additional_result_info = {"dataset": data_file_name,
                                  "cvi": cvi.get_abbrev(),
                                  "n": n, "f": f,
                                  "true_k": true_k,
                                  "run": run,
                                  "mf_set": Helper.mf_set_to_string(mf_set)
                                  }
        print(f"Running on data: {data_file_name}")

        aml_optimizer = SMACOptimizer(dataset=X, cvi=cvi,
                                      cs=cs,
                                      n_loops=80,
                                      wallclock_limit=360 * 60,
                                      # cutoff_time=10,
                                      seed=seed,
                                      limit_resources=False
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
        result_df.to_csv(result_file, index=False)
