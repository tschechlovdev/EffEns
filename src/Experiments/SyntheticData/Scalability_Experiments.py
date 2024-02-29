from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from ConsensusCS.ConsensusCS import CC_function_mapping
from EnsMetaLearning.EffEns import EffEns
from Experiments.SyntheticData import DataGeneration
from Utils.Utils import calculate_gen_info, process_result_to_dataframe, clean_up_optimizer_directory
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection


def result_already_exists(results, test_dataset, approach, run):
    if "dataset" in results.columns:
        dataset_results = results[results["dataset"] == str(test_dataset)]
        if len(dataset_results[
                   (dataset_results["Method"] == approach) & (dataset_results["run"] == run)]) > 0:
            return True
    return False


cc_functions = CC_function_mapping.keys()

if __name__ == '__main__':
    runs = list(range(1, 5))
    k_range = (2, 100)
    n_loops = 50
    execute_learning_phase = False
    path_to_mkr = "../../EffEnsMKR"

    dataset_types = DATASET_TYPES = ['varied', 'circles',
                                     'moons'
                                     ]
    n_values = [
        1000,
        1000000,
        10000,
        100000,
    ]
    n_features = 100
    n_clusters = 50

    result_file = "scalability_results.csv"
    if Path(result_file).is_file():
        results = pd.read_csv("scalability_results.csv")
    else:
        results = pd.DataFrame()

    for run in runs:
        random_state = run * 1234
        for data_type in dataset_types:
            for n in n_values:
                # Generate data
                # We actually generate only one dataset, because otherwise we overload the RAM
                # if we create all datasets at once
                datasets = DataGeneration.generate_datasets(n_values=[n],
                                                            n_feature_values=[n_features],
                                                            k_values=[n_clusters],
                                                            dataset_types=[data_type],
                                                            random_state=random_state,
                                                            noise_values=[0.0]
                                                            )

                print(f"Running on:")
                print(datasets.keys())

                for dataset_name, data in datasets.items():
                    if result_already_exists(results, dataset_name, "EffEns", run):
                        print(f"Result for EffEns on {dataset_name} for run={run} already existing!")
                        print("Skipping")
                        continue
                    if "varied" in dataset_name:
                        cvi = CVICollection.CALINSKI_HARABASZ
                    else:
                        cvi = CVICollection.DAVIES_BOULDIN

                    X = data[0]
                    y = data[1]
                    effEns = EffEns(k_range=k_range,
                                    random_state=random_state,
                                    path_to_mkr=path_to_mkr)

                    CF_MODEL_FILE = Path(path_to_mkr) / f'model/cf_model_{run}.joblib'
                    EG_MODEL_FILE = Path(path_to_mkr) / f"model/k_pred_ens_run{run}.joblib"
                    CFM = load(CF_MODEL_FILE)
                    EGM = load(EG_MODEL_FILE)

                    print(X.shape)
                    effEns_result, add_info = effEns.apply_ensemble_clustering(X, cvi=cvi,
                                                                               n_loops=n_loops)

                    selected_ens = effEns.get_ensemble()
                    k_values_ens = [len(np.unique(x)) for x in selected_ens.transpose()]

                    # Parse Results
                    gen_info = calculate_gen_info(X, y, effEns, cvi)
                    add_info.update(gen_info)
                    add_info.update({"dataset": dataset_name,
                                     "Method": "EffEns",
                                     "run": run,
                                     "rs": random_state})

                    result = process_result_to_dataframe(effEns_result,
                                                         add_info,
                                                         ground_truth_clustering=y)
                    # Cleanup
                    clean_up_optimizer_directory(effEns_result)
                    print(result[["iteration", "config", "k_pred", "Best ARI"]])

                    # Store results
                    results = pd.concat([results, result], ignore_index=True)
                    print(result[["iteration", "config", "best config", "k_pred", "Best ARI"]])

                    result["ens"] = [k_values_ens for i in range(len(result))]
                    results = pd.concat([results, result])

                    print(results)
                    print(f"Storing result to {str(result_file)}")
                    results.to_csv(result_file, index=False)
