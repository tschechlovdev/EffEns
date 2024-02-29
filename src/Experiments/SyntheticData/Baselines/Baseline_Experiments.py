import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS

from ConsensusCS.ConsensusCS import build_consensus_cs, CC_FUNCTIONS
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from EnsOptimizer.EnsembleSelection import SelectionStrategies
from Experiments.SyntheticData import DataGeneration
from Experiments.SyntheticData.Baselines.CASBaseline import generate_ensemble
from Utils.RAMManager import memory
from Utils.Utils import clean_up_optimizer_directory, process_result_to_dataframe, get_n_from_dataset
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.ClusteringCS import ClusteringCS
from automlclustering.ClusteringCS.ClusteringCS import CONFIG_SPACE_MAPPING, KMEANS_SPACE, DBSCAN_SPACE, \
    ALL_ALGOS_SPACE, PARTITIONAL_SPACE, build_config_space
from automlclustering.MetaLearning import MetaFeatureExtractor
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer


def get_similar_datasets(test_mfs_values, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(training_mfs.iloc[:, :-1].to_numpy())
    distances, indices = nbrs.kneighbors(test_mfs_values.reshape(1, -1))
    print(indices)
    return training_mfs.iloc[indices[0]]["dataset"]


runs = list(range(1, 5))
k_range = (2, 100)
n_loops = 70
execute_learning_phase = False
path_to_mkr = "../../EffEnsMKR"
cs_all = ClusteringCS.build_all_algos_space(k_range=k_range)
cs_part = ClusteringCS.build_partitional_config_space(k_range=k_range)
cs_mapping = {cs_name: cs_function
              for cs_name, cs_function in CONFIG_SPACE_MAPPING.items()
              if cs_name != KMEANS_SPACE
              and cs_name != DBSCAN_SPACE  # Problems with memory for n=50k and 'large' epsilon ...
              }

result_file = "baseline_results_new.csv"

if Path(result_file).is_file():
    results = pd.read_csv(result_file)
else:
    results = pd.DataFrame()

eval_configs = pd.read_csv("evaluated_configs.csv")
best_eval_configs = eval_configs.loc[eval_configs.groupby(["dataset"])["AMI"].idxmin()]
best_eval_configs["algorithm"] = best_eval_configs.apply(
    lambda x: ast.literal_eval(x["config"])["algorithm"],
    axis="columns")
datasets = DataGeneration.generate_datasets(n_values=[1000, 10000, 50000])

effens_result = pd.read_csv("../results/synthetic_cf_es_m_pred_run1.csv")
print(sorted(effens_result["dataset"].unique()))
print(len(effens_result["dataset"].unique()))

test_datasets = {data_name: value for data_name, value in datasets.items()
                 if data_name in effens_result["dataset"].unique()}

# Make sure these are the same datasets as used for EffEns
print(sorted(list(test_datasets.keys())))
print(len(test_datasets.keys()))
assert (len(test_datasets.keys()) == len(effens_result["dataset"].unique()))
assert (sorted(list(test_datasets.keys())) == sorted(effens_result["dataset"].unique()))


@memory(percentage=1.5)
def run_optics(X):
    optics = OPTICS()
    y_pred = optics.fit_predict(X)
    return y_pred


for run in runs:
    random_state = run * 1234

    test_datasets = {k: test_datasets[k] for k
                     in sorted(list(test_datasets.keys()),
                               key=lambda dataset_name: int(get_n_from_dataset(dataset_name)),
                               reverse=True)}
    print(test_datasets.keys())

    cvi = CVICollection.CALINSKI_HARABASZ
    for dataset_name, data in test_datasets.items():

        if "circles" in dataset_name or "moons" in dataset_name:
            cvi = CVICollection.COP_SCORE

        for method in [
            "AS->HPO",
            "AutoClust",
            "aml4c(k)",
            "aml4c(A)",
            "optics",
            "AEC",
        ]:
            print(f"Running Method: {method}")

            print(f"Running on:")
            print(dataset_name)
            if len(results) > 0:
                existing_result = results[
                    (results["Method"] == method) & ((results["run"] == run) | (results["run"].isnull())) & (
                            results["dataset"] == dataset_name)]
                if len(existing_result) > 0:
                    print("Result already existing")
                    print(existing_result)
                    print("Continue!")
                    continue

            X = data[0]
            y = data[1]
            n = X.shape[0]

            additional_result_info = {"dataset": dataset_name,
                                      "cvi": cvi.get_abbrev(),
                                      "n": n,
                                      "run": run
                                      }

            if method == "optics":
                if run > 1:
                    # non-deterministic, so skip for further runs
                    continue
                error = None
                start = time.time()
                try:
                    y_pred = run_optics(X)
                except MemoryError as error:
                    print("Memory error occured!!!")
                    print(str(error))
                    y_pred = np.ones(X.shape[0])

                runtime = time.time() - start
                print("Calculating NMI score")
                ami_score = CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, labels=y_pred,
                                                                    true_labels=y)
                print(f"Finished NMI score with score {ami_score}")

                print("Calculating ARI score")
                ari_score = CVICollection.ADJUSTED_RAND.score_cvi(data=None,
                                                                  labels=y_pred,
                                                                  true_labels=y)
                print(f"Finished ARI score with score {ari_score}")

                result = {"Method": [method], "iteration": [n_loops],
                          "wallclock time": [runtime], "Best NMI": [ami_score],
                          "Best ARI": [ari_score],
                          "error": [error],
                          "dataset": [dataset_name]
                          }
                result_df = pd.DataFrame.from_dict(result)
                results = pd.concat([results, result_df], ignore_index=True)
                results.to_csv(result_file, index=False)

                # Following code is only for other baselines
                continue

            elif method == "AEC":
                print(f"Running cf={'all'} on {dataset_name}")

                consensus_cs = build_consensus_cs(k_range=k_range,
                                                  cc_functions=CC_FUNCTIONS,
                                                  max_ensemble_size=50)
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
                                                  ensemble=ens,
                                                  wallclock_limit=360 * 60,
                                                  limit_resources=False
                                                  )
                consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_loops,
                                                                 k_range=k_range)
                # store result
                additional_result_info = {"dataset": dataset_name, "n": X.shape[0],
                                          "cvi": cvi.get_abbrev(), "Method": method,
                                          "run": run,
                                          "es_strategy": SelectionStrategies.cluster_and_select.value,
                                          "cf": "all", "gen_time": gen_time}
                consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                                  additional_result_info,
                                                                  ground_truth_clustering=y)
                results = pd.concat([results,
                                     consensus_result_df], ignore_index=True)
                results.to_csv(result_file, index=False)

                # Following code is only for other baselines
                continue

            elif method == "aml4c(A)":
                cs = ALL_ALGOS_SPACE
                cs = CONFIG_SPACE_MAPPING[cs](k_range=k_range, X_shape=X.shape)
            elif method == "aml4c(k)":
                cs = PARTITIONAL_SPACE
                cs = CONFIG_SPACE_MAPPING[cs](k_range=k_range)

            elif method == "AutoClust" or method == "AS->HPO":
                if method == "AutoClust":
                    mf_set = "autoclust"
                    training_mfs = pd.read_csv("autoclust_metafeatures.csv")

                else:
                    mf_set = ["statistical", "general"]
                    training_mfs = pd.read_csv("statistical_general_meta_features.csv")

                mfs, test_mfs_values = MetaFeatureExtractor.extract_meta_features(X, mf_set)

                training_mfs = training_mfs[~training_mfs["dataset"].isin(
                        list(test_datasets.keys()))]

                print(training_mfs)
                test_mfs_values = np.nan_to_num(test_mfs_values, nan=0)
                print(f"Similar Datasets:")
                similar_datasets = get_similar_datasets(test_mfs_values, k=10)
                best_conf_similar_datasets = best_eval_configs[
                    best_eval_configs["dataset"].isin(similar_datasets)]
                print(best_conf_similar_datasets[["dataset", "algorithm", "AMI"]])
                print(best_conf_similar_datasets["algorithm"].value_counts().idxmax())
                print(best_conf_similar_datasets["algorithm"].mode())

                best_algorithm = best_conf_similar_datasets["algorithm"].value_counts().idxmax()
                print(best_algorithm)

                cs = build_config_space(clustering_algorithms=[best_algorithm],
                                        k_range=k_range,
                                        X_shape=X.shape)
                print(cs)

            else:
                print(f"Unknown method found: {method}")
                print("continue")
                continue

            # Same optimizer for AML4C, AutoClust, AS->HPO, only different CS
            optimizer = SMACOptimizer(dataset=X, cvi=cvi,
                                      cs=cs,
                                      n_loops=n_loops,
                                      wallclock_limit=360 * 60,
                                      # cutoff_time=10,
                                      seed=random_state,
                                      limit_resources=False
                                      )

            # Same code for AutoML Approaches
            optimizer.optimize()
            result_df = process_result_to_dataframe(optimizer,
                                                    additional_result_info,
                                                    ground_truth_clustering=y)
            clean_up_optimizer_directory(optimizer)
            result_df["Method"] = method
            results = pd.concat([results, result_df], ignore_index=True)
            results.to_csv(result_file, index=False)
