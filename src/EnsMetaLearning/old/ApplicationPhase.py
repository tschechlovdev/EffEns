# New Dataset
import ast
import os.path
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from ClusterValidityIndices import CVIHandler
from ConsensusCS import ConsensusCS
from ConsensusCS.ConsensusCS import build_consensus_cs
from EnsMetaLearning import MKR
from EnsMetaLearning.Helper import process_result_to_dataframe, clean_up_optimizer_directory
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from MetaLearning import MetaFeatureExtractor


# Parameters:
# General: n_loops, CVI,
# Meta-learning: meta-feature-set, use_classifier, (reduce_CS, n_warmstarts, n_similar_datasets)


def _extract_meta_features(X, mf_set):
    return MetaFeatureExtractor.extract_meta_features(X, mf_set)[1]


def _get_similar_datasets(mf_new, path_to_mkr_meta_features, d_name, n_similar_datasets):
    mf_df = pd.read_csv(path_to_mkr_meta_features, index_col=None)
    mf_df.sort_values("dataset")

    mf_df = mf_df[mf_df["dataset"] != d_name]
    datasets = mf_df["dataset"].values
    mf_learning_values = mf_df.drop("dataset", axis=1).to_numpy()
    print(mf_learning_values.shape)
    print(mf_new.shape)

    knn = NearestNeighbors(n_neighbors=n_similar_datasets)
    knn.fit(mf_learning_values)
    distances, neighbors = knn.kneighbors(mf_new.reshape(1, -1), return_distance=True)
    distances = distances[0]
    neighbors = neighbors[0]
    print(f"dataset for query: {d_name}")
    print(f"Most-similar datasets ({n_similar_datasets}): {datasets[neighbors]}")

    return datasets[neighbors], mf_learning_values


def _get_eval_ensemble_similar_datasets(d_name, similar_datasets):
    # read evaluated ensembles
    eval_ens = pd.read_csv(MetaKnowledgeRepository.path_to_eval_ens)

    # do some preprocessing/filtering stuff
    eval_ens["n"] = eval_ens["dataset"].apply(lambda x: x.split("-")[2].split("=")[-1]).astype(int)
    eval_ens = eval_ens[eval_ens["n"] <= 50000]
    eval_ens = eval_ens.sort_values("dataset")
    eval_ens = eval_ens[eval_ens["dataset"] != d_name]

    # assign columns to have more detailed information
    eval_ens["cc_function"] = eval_ens["config"].apply(lambda x: ast.literal_eval(x)["cc_function"])
    eval_ens["k"] = eval_ens["config"].apply(lambda x: ast.literal_eval(x)["k"])
    eval_ens["m"] = eval_ens["config"].apply(lambda x: ast.literal_eval(x)["m"])

    # get evaluated ensembles for similar datasets
    d_s_evals = eval_ens[eval_ens["dataset"].isin(similar_datasets)]
    d_s_evals = d_s_evals.sort_values("ARI")
    d_s_evals = d_s_evals.drop_duplicates(["cc_function", "k"], keep="first")
    return eval_ens, d_s_evals


def train_classifier_ensemble(eval_ens, mf_values, k_range):
    # TODO: Best ARI! Das ist nur der ARI für bestimmte Konfigurationen ..
    eval_ens["Best ARI"] = eval_ens.groupby(["dataset", "m"])["ARI"].transform("min")

    ensemble_info = eval_ens[["dataset", "ARI", "Ensemble", "m"]]
    eval_ens["Best ARI"] = eval_ens.groupby(["dataset", "m"])["ARI"].transform("min")
    k_in_ensemble = {k: [] for k in range(k_range[0], k_range[1] + 1)}

    # Generate n X len(k_range) matrix -> '1' if k value is in the ensemble otherwise '0'
    for j in range(ensemble_info.shape[0]):
        row = ensemble_info.iloc[j, :]
        k_values = ast.literal_eval(row["Ensemble"])

        for k in range(k_range[0], k_range[1] + 1):
            if k in k_values:
                k_in_ensemble[k].append(1)
            else:
                k_in_ensemble[k].append(0)

    multi_label_targets = pd.DataFrame(k_in_ensemble).to_numpy()

    rf = RandomForestClassifier(n_estimators=200, random_state=1234,
                                # min_samples_leaf=10 --> Leads to m= 0 in any case
                                )
    rf.fit(mf_values, multi_label_targets)

    return rf


def generate_ensemble(ensemble_k_values, X_new, cvi, y_new):
    ens = np.zeros((len(ensemble_k_values), X_new.shape[0]))
    # Generate Ensemble based on meta-knowledge
    best_gen_cvi = np.infty
    best_gen_ari = np.infty

    gen_start = time.time()
    for i, k in enumerate(ensemble_k_values):
        km = KMeans(n_clusters=k, n_init=1, max_iter=100)
        y_k = km.fit_predict(X_new)
        cvi_value = cvi.score_cvi(X_new, y_k)
        if cvi_value < best_gen_cvi:
            best_gen_cvi = cvi_value
            best_gen_ari = CVIHandler.CVICollection.ADJUSTED_RAND.score_cvi(data=None,
                                                                            labels=y_k,
                                                                            true_labels=y_new)
        ens[i, :] = y_k
    gen_time = time.time() - gen_start
    best_m = ens.shape[0]
    ens = ens.transpose()
    generation_info = {"m": best_m, "gen_time": gen_time, "gen_cvi": best_gen_cvi, "gen_ari": best_gen_ari}
    return ens, generation_info


def run_ensemble(X_new: np.array, y_new: np.array, d_name: str, n_loops: int, cvi: CVIHandler.CVI,
                 mf_set=MetaFeatureExtractor.meta_feature_sets[2],
                 use_classifier=True,
                 reduce_cs=False,
                 n_warmstarts=0,
                 n_similar_datasets=1,
                 path_to_mkr_meta_features=MetaKnowledgeRepository.path_to_mkr_meta_features,
                 k_range=(2, 101),
                 cc_functions=ConsensusCS.CC_FUNCTIONS,
                 application_mode="Similar Dataset") -> pd.DataFrame:
    warmstart_configs = None
    actually_used_classifer = use_classifier

    # 1) Extract Meta-features
    mf_new = _extract_meta_features(X_new, mf_set)
    print(mf_new)

    # 2) Find similar datasets
    similar_datasets, mf_values = _get_similar_datasets(mf_new, path_to_mkr_meta_features,
                                                        d_name, n_similar_datasets)

    # 3) Retrieve evaluated_ensemble
    eval_ens, eval_ens_ds = _get_eval_ensemble_similar_datasets(d_name, similar_datasets)

    # 4) Train classifier
    if use_classifier:
        classif = train_classifier_ensemble(eval_ens, mf_values, k_range)

        # 4.2) Predict ensemble
        y_pred = classif.predict(mf_new.reshape(1, -1))[0]
        ensemble_k_values = [k + k_range[0] for k, value in enumerate(y_pred) if value == 1]

    # TODO: It might happen that classifier does not put any result into the ensemble ...
    # Maybe wrong meta-features?

    if not use_classifier or not ensemble_k_values or len(ensemble_k_values) == 1:
        actually_used_classifer = False
        # Use best Ensemble from most-similar dataset
        ensemble_k_values = ast.literal_eval(eval_ens_ds["Ensemble"].values[0])
    print(ensemble_k_values)
    # 5) Generate ensemble
    ens, gen_info = generate_ensemble(ensemble_k_values, X_new, cvi, y_new)

    # 6) Run consensus optimization

    # 6.1) Warmstarting
    if n_warmstarts > 0:
        warmstart_configs = eval_ens_ds['config'].values[0:n_warmstarts]

    # 6.2) Reduce CS
    if reduce_cs:
        best_cc_function = eval_ens_ds["cc_function"].values[0]
        print(f"Using cc-function: {best_cc_function}")
        cc_functions = [ConsensusCS.CC_function_mapping[best_cc_function]]

    # 6.3) Build consensus cs
    consensus_cs = build_consensus_cs(k_range=k_range,
                                      cc_functions=cc_functions,
                                      default_ensemble_size=len(ensemble_k_values),
                                      max_ensemble_size=None)

    # 6.4) Optimization
    ens_optimizer = EnsembleOptimizer(dataset=X_new,
                                      generation_cvi=cvi,
                                      # true_labels=y,
                                      cs_generation=consensus_cs,
                                      cs_consensus=consensus_cs,
                                      seed=10)
    ens_optimizer.ensemble = ens
    ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=k_range)

    # 7.) Parsing the results + cleanup
    additional_result_info = ({"dataset": d_name,
                               "similar dataset": similar_datasets,
                               "cvi": cvi.get_abbrev(),
                               "used_classifier": actually_used_classifer,
                               "reduce_cs": reduce_cs,
                               "classifier": use_classifier,
                               "m": len(ensemble_k_values)})
    additional_result_info.update(gen_info)
    result = process_result_to_dataframe(ens_optimizer.consensus_optimizer,
                                         additional_result_info,
                                         ground_truth_clustering=y_new)
    clean_up_optimizer_directory(ens_optimizer.consensus_optimizer)

    return result


if __name__ == '__main__':
    import re

    # datasets = DataGeneration.generate_datasets(n_values=[1000, 5000, 10000, 50000,
    #                                                       #70000,
    #                                                       #100000,
    #                                                       # 500000, 1000000
    #                                                       ])
    # mf_per_dataset = {d_name: MetaFeatureExtractor.extract_meta_features(data[0], mf_set=["statistical", "general"],)[1]
    #                     for d_name, data in datasets.items()}
    #
    # df = pd.DataFrame(data=mf_per_dataset)
    # print(df.T)
    # df = df.T
    # df["dataset"] = df.index
    # df.to_csv("meta_features.csv",  index=None)

    all_results = pd.DataFrame()
    real_world_path = Path("/volume/datasets/real_world/")

    result_file_name = "eval_classif_cvi.csv"
    if os.path.isfile(result_file_name):
        all_results = pd.read_csv("eval_classif_cvi.csv")
    else:
        all_results = pd.DataFrame()

    for d_name in [file for file in real_world_path.glob('**/*') if file.is_file()]:
        n = int(re.search("_n\d\d\d\d?\d?_", str(d_name)).group().replace("_n", "").replace("_", ""))
        if n > 20000:
            continue
        data = pd.read_csv(real_world_path / d_name, header=None)
        print(data)
        X_new = data.iloc[:, :-1].to_numpy()
        y_new = data.iloc[:, -1].to_numpy()

        for cvi in [CVIHandler.CVICollection.CALINSKI_HARABASZ,
                    CVIHandler.CVICollection.DAVIES_BOULDIN,
                    CVIHandler.CVICollection.DENSITY_BASED_VALIDATION
                    ]:
            for use_classifier in [True, False]:
                for reduce_cs in [True, False]:
                    for run in range(3):
                        result = run_ensemble(X_new,
                                              y_new,
                                              n_loops=100,
                                              cvi=cvi,
                                              d_name=str(d_name),
                                              mf_set=MetaFeatureExtractor.meta_feature_sets[5],
                                              reduce_cs=reduce_cs,
                                              use_classifier=use_classifier)

                        result["run"] = run
                        all_results = pd.concat([all_results, result])
                        all_results.to_csv("eval_classif_cvi.csv", index=False)
