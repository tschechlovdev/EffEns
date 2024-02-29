import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS.ConsensusCS import CC_function_mapping
from EnsMetaLearning.EffEns import EffEns
from Utils.Utils import get_type_from_dataset, get_noise_from_dataset, process_result_to_dataframe, \
    clean_up_optimizer_directory, calculate_gen_info, path_to_eval_results
from Experiments.SyntheticData import DataGeneration


def result_already_exists(results, test_dataset, approach, cvi):
    if "dataset" in results.columns:
        dataset_results = results[results["dataset"] == str(test_dataset)]
        if len(dataset_results[
                   (dataset_results["Method"] == approach) & (dataset_results["cvi"] == cvi.get_abbrev())]) > 0:
            return True
    return False


cc_functions = CC_function_mapping.keys()


def get_mf_scaler(path_to_mkr, training_datasets_df):
    mf_path = "meta_features.csv"
    mfs_df = pd.read_csv(Path(path_to_mkr) / mf_path)
    mfs_df = mfs_df[mfs_df["dataset"].isin(dataset_names)]
    mfs_df = mfs_df.sort_values("dataset")
    mfs_train = mfs_df.iloc[training_datasets_df.index]
    X_metafeatures_train = mfs_train.drop("dataset", axis=1).to_numpy()
    mf_scaler = StandardScaler()
    mf_scaler.fit(X_metafeatures_train)
    return mf_scaler


if __name__ == '__main__':
    runs = list(range(1, 7))
    k_range = (2, 100)
    n_loops = 70
    execute_learning_phase = False
    path_to_mkr = "../../EffEnsMKR"

    # Generate data
    datasets = DataGeneration.generate_datasets(n_values=[1000,  # 5000,
                                                          10000, 50000,
                                                          ])

    X = [data[0] for _, data in datasets.items()]
    y_labels = [data[1] for _, data in datasets.items()]

    dataset_names = [d_name for d_name, _ in datasets.items()]
    dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
    dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
    dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)
    dataset_df = dataset_df.sort_values("dataset")
    training_datasets_df, test_datasets_df = train_test_split(dataset_df,
                                                              stratify=dataset_df[["type", "noise"]],
                                                              train_size=0.7,
                                                              random_state=1234)
    mf_scaler = get_mf_scaler(path_to_mkr, training_datasets_df)

    for run in runs:
        random_state = run * 1234

        effEns = EffEns(k_range=k_range,
                        random_state=random_state,
                        path_to_mkr=path_to_mkr)
        if execute_learning_phase:
            pass
            # TODO: Execute learning phase
            # effEns.run_learning_phase(dataset_names=training_datasets_df["dataset"].unique(),
            #                          )
        else:
            # Learning phase should already be executed, so just load models
            CF_MODEL_FILE = Path(path_to_mkr) / f'model/cf_model_{run}.joblib'
            EG_MODEL_FILE = Path(path_to_mkr) / f"model/k_pred_ens_run{run}.joblib"
            CFM = load(CF_MODEL_FILE)
            EGM = load(EG_MODEL_FILE)

            effEns.EGM = EGM
            effEns.CFM = CFM
            effEns.mf_scaler = mf_scaler

        ##############################################################
        ##############################################################
        ##### Application Phase
        result_file = path_to_eval_results / "synthetic"
        result_file.mkdir(parents=True, exist_ok=True)
        result_file = result_file / f"synthetic_cf_es_m_pred_run{run}.csv"

        if os.path.isfile(result_file):
            results = pd.read_csv(result_file, index_col=0)
        else:
            results = pd.DataFrame()

        results = pd.DataFrame()

        print("test datasets:")
        print(test_datasets_df["dataset"].unique())

        for test_dataset in test_datasets_df["dataset"].values:
            dataset_index = dataset_names.index(test_dataset)
            X_test = X[dataset_index]
            y_test = y_labels[dataset_index]

            if "gaussian" in test_dataset or "varied" in test_dataset:
                cvi = CVICollection.CALINSKI_HARABASZ
            else:
                cvi = CVICollection.COP_SCORE

            for approach in [
                "Model CF->k-values",
                "Model CF->k-values(2,100)"
            ]:

                if "(2,100)" in approach:
                    use_k_heuristic = (2, 100)
                else:
                    use_k_heuristic = ()

                print("-------------------------")
                print(f"Running approach: {approach}")

                if result_already_exists(results, test_dataset, approach, cvi):
                    print(f"Continue - {approach} for {test_dataset} already exists!")
                    continue

                effEns_result, add_info = effEns.apply_ensemble_clustering(X=X_test, cvi=cvi,
                                                                           n_loops=n_loops,
                                                                           ensemble_k_heuristic=use_k_heuristic)
                selected_ens = effEns.get_ensemble()
                k_values_ens = [len(np.unique(x)) for x in selected_ens.transpose()]

                # Parse Results
                gen_info = calculate_gen_info(X_test, y_test, effEns, cvi)
                add_info.update(gen_info)
                add_info.update({"dataset": test_dataset,
                                 "Method": approach,
                                 "run": run,
                                 "rs": random_state})

                result = process_result_to_dataframe(effEns_result,
                                                     add_info,
                                                     ground_truth_clustering=y_test)
                # Cleanup
                clean_up_optimizer_directory(effEns_result)
                print(result[["iteration", "config", "k_pred", "Best ARI"]])

                # Store results
                results = pd.concat([results, result], ignore_index=True)
                print(result[["iteration", "config", "best config", "k_pred", "Best ARI"]])

                result["ens"] = [k_values_ens for i in range(len(result))]
                results = pd.concat([results, result])

                print(f"Storing result to {str(result_file)}")
                results.to_csv(result_file)
