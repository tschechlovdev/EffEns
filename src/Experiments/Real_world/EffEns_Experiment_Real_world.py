import os.path
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS.ConsensusCS import CC_function_mapping
from EnsMetaLearning.EffEns import EffEns
from Utils.Utils import process_result_to_dataframe, \
    clean_up_optimizer_directory, get_n_from_real_world_data, calculate_gen_info
from EnsMetaLearning import MetaFeatureExtractor
from Utils.Utils import path_to_eval_results

warnings.filterwarnings(action="ignore")


runs = list(range(10))
n_estimators = 1000
k_range = (2, 100)
n_loops = 70
path_to_mkr = "../../EffEnsMKR"

for run in runs:
    random_state = run * 1234
    cc_functions = CC_function_mapping.keys()

    ##############################################################
    ##############################################################
    ##### Application Phase
    #result_file_name = f"/home/ubuntu/automated_consensus_clustering/automated_consensus/src/EnsMetaLearning/real_world_cf_es_m_pred_run{run}.csv"
    result_file_name = path_to_eval_results / "real_world"
    result_file_name.mkdir(parents=True, exist_ok=True)
    result_file_name = result_file_name / f"real_world_cf_es_m_pred_run{run}.csv"

    if os.path.isfile(result_file_name):
        print(f"Found existing result file {result_file_name}")
        print("Skipping existing results if possible")
        results = pd.read_csv(result_file_name)
        print(results)
    else:
        print("No results - running all experiments")
        results = pd.DataFrame()

    real_world_path = Path("/volume/datasets/real_world/")
    test_dataset_names = [file for file in real_world_path.glob('**/*') if file.is_file()]

    test_dataset_names = sorted(test_dataset_names, key=get_n_from_real_world_data  # , reverse=True
                                )

    print("test datasets:")
    print(test_dataset_names)

    mf_set = MetaFeatureExtractor.meta_feature_sets[5]
    best_cvi_file = "best_cvi_real_world_data.csv"

    if os.path.isfile(best_cvi_file):
        print("File exists")
        best_cvi_df = pd.read_csv(best_cvi_file)
    else:
        print("File not exists")

    for approach in ["EffEns"]:

        print("-------------------------")
        print(f"Running approach: {approach}")
        effEns = EffEns(k_range=k_range, random_state=random_state, path_to_mkr=path_to_mkr)

        for test_dataset in test_dataset_names:

            print(real_world_path / test_dataset)
            data = pd.read_csv(test_dataset, header=None, index_col=None)
            print(f"Running on dataset {test_dataset}")
            X_new = data.iloc[:, :-1].to_numpy()
            y_new = data.iloc[:, -1].to_numpy()

            print("Possible CVIs:")
            print(best_cvi_df[best_cvi_df["dataset"] == str(test_dataset).replace("/volume/datasets/real_world/", "")][
                      "cvi"].values)
            cvi_abbrev = \
                best_cvi_df[best_cvi_df["dataset"] == str(test_dataset).replace("/volume/datasets/real_world/", "")][
                    "cvi"].values[
                    0]
            cvis = [CVICollection.get_cvi_by_abbrev(cvi_abbrev)]

            for cvi in cvis:

                ################################################################################
                ########################### Apply EffEns #######################################
                effEns_result, additional_result_info = effEns.apply_ensemble_clustering(X=X_new, cvi=cvi,
                                                                                         n_loops=5)

                # Parse Results
                gen_info = calculate_gen_info(X_new, y_new, effEns, cvi)
                additional_result_info.update(gen_info)
                additional_result_info.update({"dataset": test_dataset,
                                               "Method": approach,
                                               "run": run,
                                               "rs": random_state})

                result = process_result_to_dataframe(effEns_result,
                                                     additional_result_info,
                                                     ground_truth_clustering=y_new)
                # Cleanup
                clean_up_optimizer_directory(effEns_result)
                print(result[["iteration", "config", "k_pred", "Best ARI"]])

                # Store results
                results = pd.concat([results, result], ignore_index=True)
                results.to_csv(result_file_name, index=False)
