import ast
import os.path
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import EnsOptimizer.cas
from ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS.ConsensusCS import CC_function_mapping, CC_FUNCTIONS, build_consensus_cs
from EnsMetaLearning.Helper import get_type_from_dataset, get_noise_from_dataset, process_result_to_dataframe, \
    clean_up_optimizer_directory
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from EnsOptimizer.EnsembleSelection import SelectionStrategies
from Experiments.SyntheticData import DataGeneration


def train_cf_classifier(X_metafeatures_train, y_cf_train, random_state=1234):
    clf = RandomForestClassifier(n_estimators=1000,
                                 random_state=random_state)
    clf.fit(X_metafeatures_train, y_cf_train)
    return clf


def _append_dummies(df_train, y_train):
    df_train["tmp"] = y_train

    df_train = df_train.join(pd.get_dummies(df_train["tmp"])).drop("tmp", axis=1)
    # Corner Case for CC Functions
    if "MCLA" in y_train:
        # Maybe one of the CFs is not predicted -> We get one column less for one-hot encoding
        for cc_function in cc_functions:
            if cc_function not in df_train.columns:
                df_train[cc_function] = 0
    return df_train


def _prepare_k_ensemble_training_data(df_mfs_train, best_eval_ens_training, append_cfs=True):
    k_range = (2, 101)
    k_in_ensemble = {k: [] for k in range(k_range[0], k_range[1] + 1)}

    # Generate n X len(k_range) matrix -> '1' if k value is in the ensemble otherwise '0'
    for j in range(best_eval_ens_training.shape[0]):
        row = best_eval_ens_training.iloc[j, :]
        k_values = ast.literal_eval(row["Ensemble"])

        for k in range(k_range[0], k_range[1] + 1):
            if k in k_values:
                k_in_ensemble[k].append(1)
            else:
                k_in_ensemble[k].append(0)

    multi_label_targets = pd.DataFrame(k_in_ensemble)

    if append_cfs:
        df_mfs_cf_train = _append_dummies(df_mfs_train, best_eval_ens_training["cc_function"].values)
    else:
        df_mfs_cf_train = df_mfs_train
    return df_mfs_cf_train, multi_label_targets


def generate_ensemble(ensemble_k_values, X_new, cvi, y_new, random_state=1234):
    ens = np.zeros((len(ensemble_k_values), X_new.shape[0]))
    best_gen_cvi = np.infty
    best_gen_ari = np.infty
    best_gen_nmi = np.infty
    best_k_gen = 2

    cvi_scores = []
    gen_start = time.time()

    for i, k in enumerate(ensemble_k_values):
        print(f"Running k-Means with k={k}")
        km_time = time.time()
        km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=random_state)
        y_k = km.fit_predict(X_new)
        cvi_value = cvi.score_cvi(data=X_new, labels=y_k, true_labels=y_new)
        cvi_scores.append(cvi_value)
        print(cvi_value)

        if cvi_value < best_gen_cvi:
            best_gen_cvi = cvi_value
            best_gen_ari = CVICollection.ADJUSTED_RAND.score_cvi(data=None,
                                                                 labels=y_k,
                                                                 true_labels=y_new)
            best_gen_nmi = CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, labels=y_k,
                                                                   true_labels=y_new)
            best_k_gen = k
        print(f"Finished k-Means in {time.time() - km_time}")
        sys.stdout.flush()
        # time.sleep(1)
        ens[i, :] = y_k

    gen_time = time.time() - gen_start
    # Sort ensemble based on CVI scores
    ens = np.array([e for e, _ in sorted(zip(ens, cvi_scores), key=lambda pair: pair[1])])
    # Transpose into shape (n, #k_values)
    ens = ens.transpose()
    # best_m = ens.shape[0]
    generation_info = {  # "m": best_m,
        "gen_time": gen_time, "gen_cvi": best_gen_cvi, "gen_ari": best_gen_ari,
        "gen_nmi": best_gen_nmi}
    return ens, generation_info, cvi_scores, best_k_gen


def generate_and_select_ens(es_pred, m_pred, X_test, y_test, cvi, k_values=None):
    if k_values is None:
        k_values = list(range(2, 101))

    selected_ens, gen_info, cvi_scores, best_k_gen = generate_ensemble(ensemble_k_values=k_values, X_new=X_test,
                                                                       cvi=cvi, y_new=y_test)
    selection_start = time.time()
    # select ensemble
    if es_pred == SelectionStrategies.quality.value:
        selected_ens = selected_ens[:, :m_pred]

    elif es_pred == SelectionStrategies.cluster_and_select.value:
        cas = EnsOptimizer.cas.Cas(selected_ens)
        selected_ens = cas.cluster_and_select(selected_ens, m_pred)

    else:
        raise ValueError(f"Predicted Ensemble Selection strategy {es_pred} is unknown")
    selection_time = time.time() - selection_start
    gen_info["selection_time"] = selection_time
    gen_info["gen k_pred"] = best_k_gen
    return selected_ens, gen_info


def predict_consensus_function(mf_new, cf_model):
    cf_pred = cf_model.predict(mf_new.reshape(1, -1))[0]
    print(cf_pred)
    # Assign cc functions columns
    mf_new_df = pd.DataFrame(mf_new)
    for cf in cc_functions:
        mf_new_df[cf] = 0

    # Add a one for the predicted CF
    mf_new_df[cf_pred] = 1
    return cf_pred, mf_new_df


def predict_es_m_with_models(es_model, size_m_model, mf_new_df):
    es_pred = es_model.predict(mf_new_df.to_numpy().reshape(1, -1))[0]
    print(es_pred)
    for es in ["cluster and select", "quality"]:
        mf_new_df[es] = 0
    mf_new_df[es_pred] = 1
    m_pred = size_m_model.predict(mf_new_df.to_numpy().reshape(1, -1))[0]
    return es_pred, m_pred


def get_cf_es_m_from_similar_dataset(X_metafeatures_train, mf_new, training_datasets_df, eval_ens):
    training_datasets_df = training_datasets_df.sort_values("dataset").reset_index()
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X_metafeatures_train)
    _, neighbors = knn.kneighbors(mf_new, return_distance=True)
    neighbors = neighbors
    print(neighbors)
    similar_dataset = training_datasets_df["dataset"].values[neighbors[0][0]]
    print(f"Most-similar dataset: {similar_dataset}")
    similar_data_eval_ens = eval_ens[eval_ens["dataset"] == similar_dataset]
    results = similar_data_eval_ens[similar_data_eval_ens["ARI"] == similar_data_eval_ens["ARI"].min()]
    m_pred = results["m"].values[0]
    cf_pred = results["cc_function"].values[0]
    es_pred = results["es_selection"].values[0]
    return cf_pred, es_pred, m_pred


def run_ensemble_optimization(X_test, cf_pred, selected_ens, n_loops, k_range, cvi, y_new=None, random_state=1234):
    if cf_pred:
        # Apply Optimization with selected CF and generated ensemble
        cc_functions_to_use = [CC_function_mapping[cf_pred]]
    else:
        cc_functions_to_use = [CC_function_mapping[cf.get_name()] for cf in CC_FUNCTIONS]

    # 6.3) Build consensus cs
    consensus_cs = build_consensus_cs(k_range=k_range,
                                      cc_functions=cc_functions_to_use,
                                      default_ensemble_size=selected_ens.shape[1],
                                      max_ensemble_size=None)

    # 6.4) Optimization
    ens_optimizer = EnsembleOptimizer(dataset=X_test,
                                      generation_cvi=cvi,
                                      true_labels=y_new,
                                      cs_generation=consensus_cs,
                                      cs_consensus=consensus_cs,
                                      seed=random_state)
    ens_optimizer.ensemble = selected_ens
    ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=k_range)
    return ens_optimizer.consensus_optimizer


def result_already_exists(results, test_dataset, approach, cvi):
    if "dataset" in results.columns:
        dataset_results = results[results["dataset"] == str(test_dataset)]
        if len(dataset_results[
                   (dataset_results["Method"] == approach) & (dataset_results["cvi"] == cvi.get_abbrev())]) > 0:
            return True
    return False


cc_functions = CC_function_mapping.keys()

if __name__ == '__main__':
    runs = list(range(1, 10))
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
    training_datasets_df, test_datasets_df = train_test_split(dataset_df, stratify=dataset_df[["type", "noise"]],
                                                              train_size=0.7, random_state=1234)

    for run in runs:
        random_state = run * 1234
        ### Train-test split!

        # TODO: This can definitely be refactored into some functions ...
        # LEARNING PHASE
        # Collect Meta-features
        # Collect evaluated ensemble
        # Train Models: 1) CF Model, 2) es Model, 3) m Model, 4) k-values Model (EGM)

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

        # Get Only Best Results for each dataset
        # best_eval_ensemble = eval_ens.loc[eval_ens.groupby(["dataset"])["ARI"].idxmin()][
        #    ["dataset", "ARI", "cc_function", "m", "es_selection", "type"]]
        best_eval_ensemble = pd.DataFrame()
        for dataset in eval_ens["dataset"].unique():
            dataset_eval_ens = eval_ens[eval_ens["dataset"] == dataset]
            # print(dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()])
            best_dataset_ari = dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()]
            if ("moons" in dataset or "circles" in dataset) and "MCLA" in best_dataset_ari["cc_function"]:
                best_eval_ensemble = best_eval_ensemble.append(
                    best_dataset_ari[best_dataset_ari["cc_function"] == "MCLA"])
            # Todo: I think this part is new?
            elif "ACV" in best_dataset_ari["cc_function"]:
                best_eval_ensemble = best_eval_ensemble.append(
                    best_dataset_ari[best_dataset_ari["cc_function"] == "ACV"])
            else:
                best_eval_ensemble = best_eval_ensemble.append(
                    best_dataset_ari[best_dataset_ari["runtime"] == best_dataset_ari["runtime"].min()])
        best_eval_ensemble = best_eval_ensemble.sort_values("dataset")
        ##############################################################################

        ##############################################################################
        ### Prepare Training Data
        mfs_train = mfs_df.iloc[training_datasets_df.index]
        eval_ens_train = eval_ens.iloc[training_datasets_df.index]

        best_eval_ens_training = best_eval_ensemble.iloc[training_datasets_df.index]
        y_cf_train = [cf for cf in best_eval_ens_training["cc_function"].values]
        y_es_train = [es for es in best_eval_ens_training["es_selection"].values]
        y_m_train = [m for m in best_eval_ens_training["m"].values]

        print(list(zip(mfs_train["dataset"].values, y_cf_train)))

        mfs_test = mfs_df.iloc[test_datasets_df.index]
        eval_ens_test = eval_ens.iloc[test_datasets_df.index]

        X_metafeatures_train = mfs_train.drop("dataset", axis=1).to_numpy()
        mf_scaler = StandardScaler()
        mf_scaler.fit(X_metafeatures_train)
        X_metafeatures_train = mf_scaler.transform(X_metafeatures_train)
        ##############################################################################

        ##############################################################
        # 1) Train CF model
        CF_MODEL_FILE = f'model/cf_model_{run}.joblib'
        if os.path.isfile(CF_MODEL_FILE):
            cf_model = load(CF_MODEL_FILE)
        else:
            cf_model = train_cf_classifier(X_metafeatures_train, y_cf_train, random_state=random_state)
            dump(cf_model, CF_MODEL_FILE)
        ##############################################################

        ##############################################################################
        # Prepare ES training data
        ES_MODEL_FILE = f"model/es_model_run{run}.joblib"
        if os.path.isfile(ES_MODEL_FILE):
            es_model = load(ES_MODEL_FILE)
        else:
            df_mfs_train = pd.DataFrame(X_metafeatures_train)

            df_es_train = _append_dummies(df_mfs_train, y_cf_train)
            cc_function_order = ["ABV", "ACV", "MCLA", "QMI", "MM"]

            # 2) Train ES Model
            es_model = RandomForestClassifier(n_estimators=1000,
                                              random_state=random_state)
            es_model.fit(df_es_train.to_numpy(), y_es_train)
            dump(es_model, ES_MODEL_FILE)
        ##############################################################

        M_SIZE_MODEL_FILE = f"model/m_size_model_run{run}.joblib"
        if os.path.isfile(M_SIZE_MODEL_FILE):
            size_m_model = load(M_SIZE_MODEL_FILE)
        else:

            # Prepare training data for Ensemble Size (m) Model
            df_m_train = _append_dummies(df_es_train, y_es_train)

            # 3) Train m Model
            size_m_model = RandomForestClassifier(n_estimators=1000,
                                                  random_state=random_state)
            size_m_model.fit(df_m_train.to_numpy(), y_m_train)
            dump(size_m_model, M_SIZE_MODEL_FILE)

        ##############################################################
        # 4) Train k Prediction Model
        K_PREDICTION_MODEL_FILE = f"model/k_pred_ens_run{run}.joblib"
        if os.path.isfile(K_PREDICTION_MODEL_FILE):
            ensemble_k_model = load(K_PREDICTION_MODEL_FILE)
        else:
            df_mfs_cf_train, multi_label_targets = _prepare_k_ensemble_training_data(df_mfs_train,
                                                                                     best_eval_ens_training)

            ensemble_k_model = RandomForestClassifier(n_estimators=1000, random_state=random_state,
                                                      # min_samples_leaf=10 --> Leads to m= 0 in any case
                                                      )
            ensemble_k_model.fit(df_mfs_cf_train.to_numpy(), multi_label_targets)
            dump(ensemble_k_model, K_PREDICTION_MODEL_FILE)

        ##############################################################
        ##############################################################
        ##### Application Phase
        k_range = (2, 100)
        n_loops = 70
        result_file = f"synthetic_cf_es_m_pred_run{run}.csv"

        if os.path.isfile(result_file):
            results = pd.read_csv(result_file, index_col=0)
        else:
            results = pd.DataFrame()

        # Todo: Overwrite at the moment
        results = pd.DataFrame()

        print("test datasets:")
        print(test_datasets_df["dataset"].unique())

        for test_dataset in test_datasets_df["dataset"].values:
            dataset_index = dataset_names.index(test_dataset)
            X_test = X[dataset_index]
            y_test = y_labels[dataset_index]

            if "gaussian" in test_dataset or "varied" in test_dataset:
                cvis = [CVICollection.CALINSKI_HARABASZ]
            else:
                cvis = [CVICollection.COP_SCORE,
                        # CVICollection.DAVIES_BOULDIN
                        ]

            for cvi in cvis:
                for approach in [
                    "Model CF->k-values",
                    "Model CF->k-values(2,100)"
                ]:

                    print("-------------------------")
                    print(f"Running approach: {approach}")
                    mf_new = mfs_df[mfs_df["dataset"] == test_dataset]
                    mf_new = mf_new.drop("dataset", axis=1).to_numpy()
                    mf_new = mf_scaler.transform(mf_new)

                    if result_already_exists(results, test_dataset, approach, cvi):
                        print(f"Continue - {approach} for {test_dataset} already exists!")
                        continue
                    #
                    # if "moons" not in test_dataset and "circles" not in test_dataset:
                    #     continue

                    if "CF" in approach:
                        cf_pred, mf_new_df = predict_consensus_function(mf_new, cf_model)
                        if "es->m" in approach:
                            es_pred, m_pred = predict_es_m_with_models(es_model, size_m_model, mf_new_df)
                            print(f"predicted: CF={cf_pred}, es={es_pred}, m={m_pred}")
                            selected_ens, gen_info = generate_and_select_ens(es_pred=es_pred, m_pred=m_pred,
                                                                             X_test=X_test, y_test=y_test, cvi=cvi)

                        elif "k-values" in approach:
                            # Get predictions in form 0 for k=2, 0 for k=3, 1 for k=4 etc.
                            # The k-values with a 1 should be in the ensemble
                            pred_k_values = ensemble_k_model.predict(mf_new_df.to_numpy())[0]
                            k_values_ens = [k + k_range[0] for k, value in enumerate(pred_k_values) if value == 1]
                            if "(2,100)" in approach:
                                if 2 not in k_values_ens:
                                    k_values_ens.append(2)
                                if 100 not in k_values_ens:
                                    k_values_ens.append(100)

                            if len(k_values_ens) == 0:
                                k_values_ens.extend([2, 100])
                            selected_ens, gen_info, cvi_scores, best_k_gen = generate_ensemble(
                                ensemble_k_values=k_values_ens,
                                X_new=X_test,
                                y_new=y_test,
                                cvi=cvi,
                                random_state=random_state)
                            # We don't have the selection step here
                            gen_info["selection_time"] = 0

                    elif approach == "Similar Dataset":
                        cf_pred, es_pred, m_pred = get_cf_es_m_from_similar_dataset(X_metafeatures_train, mf_new,
                                                                                    training_datasets_df, eval_ens)
                        print(f"predicted: CF={cf_pred}, es={es_pred}, m={m_pred}")
                        selected_ens, gen_info = generate_and_select_ens(es_pred=es_pred, m_pred=m_pred,
                                                                         X_test=X_test, y_test=y_test, cvi=cvi)
                    else:
                        raise ValueError(f"approach {approach} unknown!")

                    print(f"Running optimization on  {test_dataset}")
                    print(f"Predicted CF: {cf_pred}")
                    k_values_in_ens = [len(np.unique(x)) for x in selected_ens.transpose()]

                    # Little Hack for ACV and ABV -> We also want k=100 in the ensemble to not get any errors!
                    # if (cf_pred == "ACV" or cf_pred == "ABV") and (
                    #         100 not in k_values_in_ens):
                    #     new_ens = np.zeros((selected_ens.shape[0], selected_ens.shape[1] + 1))
                    #     new_ens[:, :-1] = selected_ens
                    #     new_ens[:, -1] = KMeans(n_clusters=100, max_iter=10, n_init=1,
                    #                             random_state=random_state).fit_predict(
                    #         X_test
                    #     )
                    #     selected_ens = new_ens
                    k_values_ens = [len(np.unique(x)) for x in selected_ens.transpose()]
                    print(f"k values in ensemble: {k_values_ens}")
                    consensus_optimizer = run_ensemble_optimization(X_test, cf_pred, selected_ens, n_loops,
                                                                    k_range, cvi, random_state=random_state)
                    # Parsing the results + cleanup
                    additional_result_info = ({"dataset": test_dataset,
                                               # "similar dataset": similar_datasets,
                                               "cf": cf_pred,
                                               "cvi": cvi.get_abbrev(),
                                               # "used_classifier": actually_used_classifer,
                                               # "reduce_cs": reduce_cs,
                                               # "classifier": use_classifier,
                                               "Method": approach,
                                               "m": selected_ens.shape[1]}
                    )
                    additional_result_info.update(gen_info)
                    result = process_result_to_dataframe(consensus_optimizer,
                                                         additional_result_info, ground_truth_clustering=y_test)
                    clean_up_optimizer_directory(consensus_optimizer)
                    print(result[["iteration", "config", "best config", "k_pred", "Best ARI"]])
                    result["ens"] = [k_values_in_ens for i in range(len(result))]
                    results = pd.concat([results, result])

                    results.to_csv(result_file)
