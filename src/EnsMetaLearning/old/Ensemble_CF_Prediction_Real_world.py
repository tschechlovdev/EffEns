import ast
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS.ConsensusCS import CC_function_mapping
# from EnsMetaLearning.ApplicationPhase import _extract_meta_features
from EnsMetaLearning.Ensemble_CF_Prediction import train_cf_classifier, _append_dummies, \
    _prepare_k_ensemble_training_data, result_already_exists, predict_consensus_function, predict_es_m_with_models, \
    generate_and_select_ens, generate_ensemble, get_cf_es_m_from_similar_dataset, run_ensemble_optimization
from EnsMetaLearning.Helper import get_type_from_dataset, get_noise_from_dataset, process_result_to_dataframe, \
    clean_up_optimizer_directory, get_n_from_real_world_data
from Experiments.SyntheticData import DataGeneration
from MetaLearning import MetaFeatureExtractor
import warnings

warnings.filterwarnings(action="ignore")


def train_models(approach, random_state):
    # Generate data
    print("Loading Meta-Knowledge and preparing training data")
    dataset_names = list(DataGeneration.generate_datasets(n_values=[1000,  # 5000,
                                                                    10000, 50000,
                                                                    # 70000
                                                                    ]).keys())

    # X = [data[0] for _, data in datasets.items()]
    # y_labels = [data[1] for _, data in datasets.items()]

    # dataset_names = [d_name for d_name, _ in datasets.items()]
    dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
    dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
    dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)
    dataset_df = dataset_df.sort_values("dataset")
    ### Train-test split!

    # LEARNING PHASE
    # Collect Meta-features
    # Collect evaluated ensemble
    # Train Models: 1) CF Model, 2) es Model, 3) m Model, 4) k-values Model

    ##############################################################################
    ############## Get Meta-knowledge ############################################
    mf_path = "meta_features.csv"
    mfs_df = pd.read_csv(mf_path)
    mfs_df = mfs_df[mfs_df["dataset"].isin(dataset_names)]
    mfs_df = mfs_df.sort_values("dataset")

    eval_ensemble_path = "evaluated_ensemble.csv"
    eval_ens = pd.read_csv(eval_ensemble_path)
    eval_ens = eval_ens[eval_ens["dataset"].isin(dataset_names)]
    eval_ens = eval_ens.sort_values("dataset")

    eval_ens["type"] = eval_ens["dataset"].apply(get_type_from_dataset)
    eval_ens["m"] = eval_ens["Ensemble"].apply(lambda x: len(ast.literal_eval(x)))
    eval_ens["cc_function"] = eval_ens["config"].apply(lambda x: ast.literal_eval(x)["cc_function"])

    # This is indeed better for the generation, but not for our approach ...
    # However, for iris for instance, it does not improve the results, but what about the other datasets?
    if "CAS" in approach:
        eval_ens = eval_ens[eval_ens["es_selection"] == "cluster and select"]
    # Get Only Best Results for each dataset
    # best_eval_ensemble = eval_ens.loc[eval_ens.groupby(["dataset"])["ARI"].idxmin()][
    #    ["dataset", "ARI", "cc_function", "m", "es_selection", "type"]]
    best_eval_ensemble = pd.DataFrame()
    for dataset in eval_ens["dataset"].unique():
        dataset_eval_ens = eval_ens[eval_ens["dataset"] == dataset]
        # print(dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()])
        best_dataset_ari = dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()]
        if ("moons" in dataset or "circles" in dataset) and "MCLA" in best_dataset_ari["cc_function"]:
            best_eval_ensemble = best_eval_ensemble.append(best_dataset_ari[best_dataset_ari["cc_function"] == "MCLA"])
        elif "ACV" in best_dataset_ari["cc_function"]:
            best_eval_ensemble = best_eval_ensemble.append(best_dataset_ari[best_dataset_ari["cc_function"] == "ACV"])
        else:
            best_eval_ensemble = best_eval_ensemble.append(
                best_dataset_ari[best_dataset_ari["runtime"] == best_dataset_ari["runtime"].min()])
    best_eval_ensemble = best_eval_ensemble.sort_values("dataset")
    ##############################################################################

    reuse_models = False

    ##############################################################################
    ### Prepare Training Data
    y_cf_train = [cf for cf in best_eval_ensemble["cc_function"].values]
    y_es_train = [es for es in best_eval_ensemble["es_selection"].values]
    y_m_train = [m for m in best_eval_ensemble["m"].values]

    # mfs_test = mfs_df.iloc[test_datasets_df.index]
    # eval_ens_test = eval_ens.iloc[test_datasets_df.index]

    X_metafeatures_train = mfs_df.drop("dataset", axis=1).to_numpy()
    mf_scaler = StandardScaler()
    mf_scaler.fit(X_metafeatures_train)
    X_metafeatures_train = mf_scaler.transform(X_metafeatures_train)
    ##############################################################################

    ##############################################################
    # 1) Train CF model
    print("Training CF Model")
    CF_MODEL_FILE = 'model/cf_model.joblib'
    if reuse_models and os.path.isfile(CF_MODEL_FILE):
        cf_model = load(CF_MODEL_FILE)
    else:
        cf_model = train_cf_classifier(X_metafeatures_train, y_cf_train)
        dump(cf_model, 'model/cf_model.joblib')
    print("Finished")
    ##############################################################

    ##############################################################################
    # Prepare ES training data
    print()
    ES_MODEL_FILE = "model/es_model.joblib"
    df_mfs_train = pd.DataFrame(X_metafeatures_train)

    # if reuse_models and os.path.isfile(ES_MODEL_FILE):
    #     es_model = load(ES_MODEL_FILE)
    # else:
    #
    #     df_es_train = _append_dummies(df_mfs_train, y_cf_train)
    #     cc_function_order = ["ABV", "ACV", "MCLA", "QMI", "MM"]
    #
    #     # 2) Train ES Model
    #     es_model = RandomForestClassifier(n_estimators=n_estimators,
    #                                       random_state=random_state)
    #     es_model.fit(df_es_train.to_numpy(), y_es_train)
    #     dump(es_model, ES_MODEL_FILE)
    ##############################################################
    # M_SIZE_MODEL_FILE = "model/m_size_model.joblib"
    # if reuse_models and os.path.isfile(M_SIZE_MODEL_FILE):
    #     size_m_model = load(M_SIZE_MODEL_FILE)
    # else:
    #
    #     # Prepare training data for Ensemble Size (m) Model
    #     df_m_train = _append_dummies(df_es_train, y_es_train)
    #
    #     # 3) Train m Model
    #     size_m_model = RandomForestClassifier(n_estimators=n_estimators,
    #                                           random_state=random_state)
    #     size_m_model.fit(df_m_train.to_numpy(), y_m_train)
    #     dump(size_m_model, M_SIZE_MODEL_FILE)

    ##############################################################
    # 4) Train k Prediction Model

    print("Training k-values ensemble prediction model")
    K_PREDICTION_MODEL_FILE = "model/k_pred_ens.joblib"
    if reuse_models and os.path.isfile(K_PREDICTION_MODEL_FILE):
        ensemble_k_model = load(K_PREDICTION_MODEL_FILE)
    else:
        df_mfs_cf_train, multi_label_targets = _prepare_k_ensemble_training_data(df_mfs_train, best_eval_ensemble)

        ensemble_k_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                                  # min_samples_leaf=10 --> Leads to m= 0 in any case
                                                  )
        ensemble_k_model.fit(df_mfs_cf_train.to_numpy(), multi_label_targets)
        dump(ensemble_k_model, K_PREDICTION_MODEL_FILE)
    print("Finished")

    ### Test when not using CF Model
    print("Training k-values ensemble prediction model without CF as feature")
    df_mfs_train, multi_label_targets = _prepare_k_ensemble_training_data(df_mfs_train, best_eval_ensemble,
                                                                          append_cfs=False)

    df_mfs_train = df_mfs_train.drop("tmp", axis=1)
    ensemble_k_model_no_cf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                                    # min_samples_leaf=10 --> Leads to m= 0 in any case
                                                    )

    ensemble_k_model_no_cf.fit(df_mfs_train.to_numpy(), multi_label_targets)
    print("Finished")
    return mf_scaler, cf_model, ensemble_k_model_no_cf, ensemble_k_model


runs = list(range(10))
n_estimators = 1000

for run in runs:
    random_state = run * 1234
    cc_functions = CC_function_mapping.keys()

    ##############################################################
    ##############################################################
    ##### Application Phase
    k_range = (2, 100)
    n_loops = 70
    result_file_name = f"/home/ubuntu/automated_consensus_clustering/automated_consensus/src/EnsMetaLearning/real_world_cf_es_m_pred_run{run}.csv"

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
    for approach in [  # "Similar Dataset",
        # "Model CF->es->m",
        # TODO: CAS approaches are better for generation but not for us
        # We could test CAS/Quality separately ...
        # "Model CF->k-values (CAS)",
        # "Model CF->k-values(2,100) (CAS)",
        # "Model CF->k-values(2,100) (CAS) (k_gen)",
        # "No CF Model->k-values(2,100) (CAS)",
        # "No CF Model->k-values(2,100) (CAS) (k_gen)",
        #"No CF Model->k-values (CAS)",  # Also seems good, even best total results and no outliers
        # "Model CF->k-values",
        "Model CF->k-values(2,100)",  # This seems to be the best!!!!
        # "No CF Model->k-values(2,100)",
        # "No CF Model->k-values"
    ]:
        mf_scaler, cf_model, ensemble_k_model_no_cf, ensemble_k_model = train_models(approach, random_state)

        if "k_gen" in approach:
            k_range = (best_k_gen, best_k_gen)
            if "No CF Model" in approach:
                n_loops = 5
            else:
                n_loops = 1
        else:
            k_range = (2, 100)
            n_loops = 70
        print("-------------------------")
        print(f"Running approach: {approach}")

        for test_dataset in test_dataset_names:
            print(real_world_path / test_dataset)
            data = pd.read_csv(test_dataset, header=None, index_col=None)
            print(f"Running on dataset {test_dataset}")
            X_new = data.iloc[:, :-1].to_numpy()
            y_new = data.iloc[:, -1].to_numpy()

            mf_new = MetaFeatureExtractor.extract_meta_features(X_new, mf_set)[1].reshape(1, -1)
            mf_new = mf_scaler.transform(mf_new)
            # if "dermatology" not in str(test)

            print("Possible CVIs:")
            print(best_cvi_df[best_cvi_df["dataset"] == str(test_dataset).replace("/volume/datasets/real_world/", "")][
                      "cvi"].values)
            cvi_abbrev = \
                best_cvi_df[best_cvi_df["dataset"] == str(test_dataset).replace("/volume/datasets/real_world/", "")][
                    "cvi"].values[
                    0]
            cvis = [CVICollection.get_cvi_by_abbrev(cvi_abbrev)]
            if "Fashion-MNIST_n20000_f784_c10.csv" in str(test_dataset) or "letter" in str(test_dataset):
                cvis.append(CVICollection.DENSITY_BASED_VALIDATION)
                cvis.append(CVICollection.CALINSKI_HARABASZ)

            # cvis = [best_cvi_df[best_cvi_df["dataset"] == str(test_dataset).replace("/volume/datasets/real_world/", "")]["cvi"]]

            # Old Approach: Run all cvis -> We reuse the knowledge which CVIs are best suited!
            # cvis = [
            #    CVICollection.COP_SCORE, CVICollection.CALINSKI_HARABASZ, CVICollection.DAVIES_BOULDIN]

            for cvi in cvis:

                if result_already_exists(results, test_dataset, approach, cvi):
                    print(f"Continue - {approach} for {test_dataset} already exists!")
                    continue
                #
                # if "moons" not in test_dataset and "circles" not in test_dataset:
                #     continue
                if cvi.get_abbrev() == CVICollection.DENSITY_BASED_VALIDATION.get_abbrev() and X_new.shape[0] >= 20000:
                    print(f"Cannot run DBCV on data with n >= 20k!")
                    print(f"Continue for Data {test_dataset} with shape {X_new.shape}")
                    continue

                if "CF" in approach:
                    cf_pred, mf_new_df = predict_consensus_function(mf_new, cf_model)
                    # if "es->m" in approach:
                    #     es_pred, m_pred = predict_es_m_with_models(es_model, size_m_model, mf_new_df)
                    #     selected_ens, gen_info = generate_and_select_ens(es_pred=es_pred, m_pred=m_pred,
                    #                                                      X_test=X_new, y_test=y_new, cvi=cvi)

                    if "k-values" in approach:
                        # Get predictions in form 0 for k=2, 0 for k=3, 1 for k=4 etc.
                        # The k-values with a 1 should be in the ensemble
                        if "No CF Model" in approach:
                            cf_pred = None
                            pred_k_values = ensemble_k_model_no_cf.predict(mf_new)[0]
                        else:
                            pred_k_values = ensemble_k_model.predict(mf_new_df.to_numpy())[0]
                        k_values_ens = [k + k_range[0] for k, value in enumerate(pred_k_values) if value == 1]
                        if "(2,100)" in approach:
                            if 2 not in k_values_ens:
                                k_values_ens.append(2)
                            if 100 not in k_values_ens:
                                k_values_ens.append(100)
                        selected_ens, gen_info, cvi_scores, best_k_gen = generate_ensemble(
                            ensemble_k_values=k_values_ens,
                            X_new=X_new,
                            y_new=y_new, cvi=cvi, random_state=random_state)
                        # We don't have the selection step here
                        gen_info["selection_time"] = 0
                        gen_info["gen k_pred"] = best_k_gen

                # elif approach == "Similar Dataset":
                #     cf_pred, es_pred, m_pred = get_cf_es_m_from_similar_dataset(X_metafeatures_train, mf_new,
                #                                                                 dataset_df, eval_ens)
                #     print(f"predicted: CF={cf_pred}, es={es_pred}, m={m_pred}")
                #     selected_ens, gen_info = generate_and_select_ens(es_pred=es_pred, m_pred=m_pred,
                #                                                      X_test=X_new, y_test=y_new, cvi=cvi)
                else:
                    raise ValueError(f"approach {approach} unknown!")

                print(f"Running optimization on  {test_dataset}")
                print(f"Predicted CF: {cf_pred}")
                k_values_in_ens = [len(np.unique(x)) for x in selected_ens.transpose()]

                # Little Hack for ACV and ABV -> We also want k=100 in the ensemble to not get any errors!
                if (cf_pred == "ACV" or cf_pred == "ABV") and (
                        100 not in k_values_in_ens):
                    new_ens = np.zeros((selected_ens.shape[0], selected_ens.shape[1] + 1))
                    new_ens[:, :-1] = selected_ens
                    new_ens[:, -1] = KMeans(n_clusters=100, max_iter=10, n_init=1,
                                            random_state=random_state).fit_predict(X_new)
                    selected_ens = new_ens

                print(f"k values in ensemble: {[len(np.unique(x)) for x in selected_ens.transpose()]}")
                consensus_optimizer = run_ensemble_optimization(X_new, cf_pred, selected_ens, n_loops,
                                                                k_range, cvi, y_new, random_state=random_state)
                # Parsing the results + cleanup
                additional_result_info = ({"dataset": test_dataset,
                                           # "similar dataset": similar_datasets,
                                           "cf": cf_pred,
                                           "cvi": cvi.get_abbrev(),
                                           # "used_classifier": actually_used_classifer,
                                           # "reduce_cs": reduce_cs,
                                           # "classifier": use_classifier,
                                           "Method": approach,
                                           "m": selected_ens.shape[1],
                                           "run": run,
                                           "rs": random_state})
                additional_result_info.update(gen_info)
                result = process_result_to_dataframe(consensus_optimizer,
                                                     additional_result_info, ground_truth_clustering=y_new)
                clean_up_optimizer_directory(consensus_optimizer)
                print(result[["iteration", "config", "k_pred", "Best ARI"]])
                results = pd.concat([results, result], ignore_index=True)
                results.to_csv(result_file_name, index=False)
