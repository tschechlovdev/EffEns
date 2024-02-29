import ast
from pathlib import Path

import numpy as np
import pandas as pd

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.ClusteringCS.ClusteringCS import build_config_space
from Experiments.SyntheticData import DataGeneration
from automlclustering.Helper import Helper
from automlclustering.MetaLearning import MetaFeatureExtractor
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer
from Utils.Utils import get_n_from_dataset, process_result_to_dataframe, clean_up_optimizer_directory

ens_results = pd.read_csv("../../../EnsMetaLearning/synthetic_cf_es_m_pred_run2.csv")
test_datasets = ens_results["dataset"].unique()

datasets = DataGeneration.generate_datasets(n_values=[1000,  # 5000,
                                                      10000, 50000,
                                                      # 70000 --> Todo: Have to extract meta-features for them
                                                      ])

dataset_names = [name for name, data in datasets.items()]
dataset_names = sorted(dataset_names, key=lambda x: int(get_n_from_dataset(x)),  # reverse=True
                       )
print(dataset_names)

df = pd.read_csv(
    f"/home/ubuntu/automated_consensus_clustering/automated_consensus/src/EnsMetaLearning/synthetic_cf_es_m_pred_run0.csv")
test_datasets = df["dataset"].unique()

test_data_names = [test_dataset_name for test_dataset_name in
                   dataset_names if test_dataset_name in test_datasets]
test_datasets = [datasets[test_data][0] for test_data in test_data_names]
test_labels = [datasets[test_data][1] for test_data in test_data_names]

for x, y, z in zip(test_data_names, [x.shape for x in test_datasets],
                   [len(np.unique(y)) for y in test_labels]):
    print(x)
    print(y)
    print(z)
    print("-------------")

#method = f"AS -> HPO"
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
training_mfs = meta_features[~meta_features["dataset"].isin(test_data_names)]
print(training_mfs)

test_mfs = meta_features[meta_features["dataset"].isin(test_data_names)]
eval_configs = pd.read_csv("evaluated_configs.csv")
eval_configs = eval_configs[~eval_configs["dataset"].isin(test_data_names)]
best_eval_configs = eval_configs.loc[eval_configs.groupby(["dataset"])["AMI"].idxmin()]
best_eval_configs["algorithm"] = best_eval_configs.apply(
    lambda x: ast.literal_eval(x["config"])["algorithm"],
    axis="columns")

print(best_eval_configs[["dataset", "algorithm"]])


def get_similar_datasets(test_mfs_values, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(training_mfs.iloc[:, :-1].to_numpy())
    distances, indices = nbrs.kneighbors(test_mfs_values.reshape(1, -1))
    print(indices)
    return training_mfs.iloc[indices[0]]["dataset"]


for run in list(range(runs)):
    print("--------------------------------------")
    print(f"Starting run {run}")
    seed = 2 * run
    np.random.seed(seed)
    for dataset_name, X, y in zip(test_data_names, test_datasets, test_labels):
        if "type=varied" in dataset_name or "gaussian" in dataset_name:
            cvi = CVICollection.CALINSKI_HARABASZ
        else:
            cvi = CVICollection.COP_SCORE

        result_path = Path(f"../results/baselines/{method}/run_{run}/{cvi.get_abbrev()}")
        if not result_path.exists():
            result_path.mkdir(exist_ok=True, parents=True)

        result_file = result_path / (dataset_name + ".csv")

        if result_file.exists():
            print(f"Result for {result_file} already existing!")
            continue

        print("-------------")
        print("----------------")
        print(f"Running on dataset: {dataset_name}")
        mfs, test_mfs_values = MetaFeatureExtractor.extract_meta_features(X, mf_set)
        test_mfs_values = np.nan_to_num(test_mfs_values, nan=0)

        print(f"Similar Datasets:")
        similar_datasets = get_similar_datasets(test_mfs_values, k=10)
        best_conf_similar_datasets = best_eval_configs[best_eval_configs["dataset"].isin(similar_datasets)]
        print(best_conf_similar_datasets[["dataset", "algorithm", "AMI"]])
        print(best_conf_similar_datasets["algorithm"].value_counts().idxmax())
        print(best_conf_similar_datasets["algorithm"].mode())

        best_algorithm = best_conf_similar_datasets["algorithm"].value_counts().idxmax()

        cs = build_config_space(clustering_algorithms=[best_algorithm], k_range=(2, 100), X_shape=X.shape)
        print(cs)

        true_k = len(np.unique(y))
        n = X.shape[0]

        f = X.shape[1]
        noise = float(dataset_name.replace(".csv", "").split("-")[4].split("=")[-1])
        type_ = dataset_name.split("-")[0].split("=")[-1]

        additional_result_info = {"dataset": dataset_name,
                                  "cvi": cvi.get_abbrev(),
                                  "n": n, "f": f,
                                  "true_k": true_k,
                                  "noise": noise, "type": type_,
                                  "run": run,
                                  "mf_set": Helper.mf_set_to_string(mf_set)
                                  }
        print(f"Running on data: {dataset_name}")

        noise = float(dataset_name.replace(".csv", "").split("-")[4].split("=")[-1])

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
