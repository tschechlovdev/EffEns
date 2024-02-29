import ast
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from pandas.core.common import SettingWithCopyWarning

from ClusterValidityIndices.CVIHandler import CVICollection
from ClusteringCS import ClusteringCS
from MetaLearningExperiments import DataGeneration
from MetaLearning import LearningPhase
from MetaLearning.MetaFeatureExtractor import load_kdtree, query_kdtree, extract_meta_features
from Optimizer.OptimizerSMAC import SMACOptimizer
from Utils import Helper
from Utils.Helper import mf_set_to_string

mkr_path = LearningPhase.mkr_path
# define random seed
np.random.seed(1234)

shape_sets = DataGeneration.generate_datasets()
# datasets_to_use = [dataset[0] for key, dataset in shape_sets.items()]
dataset_names_to_use = list(shape_sets.keys())


def find_similar_dataset(mkr_path, meta_features, dataset_name, mf_set):
    # Load kdtree --> Used to find similar dataset more efficiently
    tree = load_kdtree(path=mkr_path / LearningPhase.meta_feature_path, mf_set=mf_set)

    # Find nearest neighbors, i.e., datasets in this case
    dists, inds = query_kdtree(meta_features, tree)
    inds = inds[0]
    # We could also use the distances, but we do not need them here as the indices are already sorted by distance
    dists = dists[0]
    # Get similar datasets in their order w.r.t. distance
    most_similar_dataset_names = [dataset_names_to_use[ind] for ind in inds]
    if dataset_name == most_similar_dataset_names[0]:
        # In the experiments of our paper, we might have the same dataset in the MKR.
        # Therefore, we do not want to use it here and use the next-similar dataset
        D_s = most_similar_dataset_names[1]
    else:
        # Get the most-similar dataset, to use more datasets the following code has to be slightly adapted
        # However, we figured out that using more less similar datasets leads to a performance decrease!
        D_s = most_similar_dataset_names[0]
    return D_s


def remove_duplicates_from_ARI_s(ARI_s):
    ARI_s["algorithm"] = assign_algorithm_column(ARI_s)
    ARI_s = ARI_s.drop_duplicates(subset="config", keep='first')
    ARI_s = ARI_s.drop_duplicates(subset=["algorithm", "ARI"], keep='first')
    ARI_s = ARI_s.drop("algorithm", axis=1)
    return ARI_s


def _get_warmstart_config_from_results(warmstart_configs):
    print(warmstart_configs)
    # the configs are saved as strings, so we need ast.literal_eval to convert them to dictionaries
    warmstart_configs = [ast.literal_eval(config_string) for config_string in warmstart_configs]
    return warmstart_configs


def assign_algorithm_column(df_with_configs):
    return df_with_configs.apply(
        lambda x: ast.literal_eval(x["config"])["algorithm"],
        axis="columns")


def predict_cvi(MF, dataset_name=None, mkr_path=mkr_path, mf_set=["statistical", "info-theory", "general"]):
    # Retrieve classification model to predict CVI
    classifier_instance = joblib.load(
        f"{mkr_path}/models/{Helper.get_model_name()}/{mf_set_to_string(mf_set)}/{dataset_name}")
    predicted_cvi = classifier_instance.predict(MF.reshape(1, -1))[0]
    predicted_cvi = CVICollection.get_cvi_by_abbrev(predicted_cvi)
    return predicted_cvi


def select_warmstart_configurations(ARI_s, n_warmstarts):
    if n_warmstarts > 0:
        warmstart_configs = ARI_s.sort_values("ARI", ascending=True)[0:n_warmstarts]
        return warmstart_configs
    else:
        return []


def define_config_space(warmstart_configs, limit_cs=True):
    if limit_cs:
        warmstart_configs["algorithm"] = assign_algorithm_column(warmstart_configs)
        algorithms = list(warmstart_configs["algorithm"].unique())
        # Use algorithms from warmstarts to build CS
        cs = ClusteringCS.build_config_space(clustering_algorithms=algorithms)

        # update warmstart configurations
        warmstart_configs = warmstart_configs["config"]
        warmstart_configs = [ast.literal_eval(config_string) for config_string in warmstart_configs]
        warmstart_configs = [Configuration(cs, config_dict) for config_dict in warmstart_configs]
    else:
        # Use default config space
        cs = ClusteringCS.build_config_space()
    return cs, warmstart_configs, algorithms


def retrieve_ARI_values_for_similar_dataset(EC, D_s):
    EC_s = EC[EC["dataset"] == D_s]
    ARI_s = EC_s[["config", "ARI"]]
    ARI_s = remove_duplicates_from_ARI_s(ARI_s)
    return ARI_s


def run_application_phase(X, mkr_path=mkr_path,
                          dataset_name=None,
                          n_warmstarts=25,
                          n_optimizer_loops=100,
                          cvi="predict",  # Otherwise, a CVI from the CVICollection
                          limit_cs=True,  # Used to reduce the configuration space based on warmstart configs
                          time_limit=120 * 60,  # Set default timeout after 2 hours of optimization
                          optimizer=SMACOptimizer,
                          mf_set=["statistical", "info-theory", "general"]
                          ):
    # Keeps track of additional Information, e.g., mf extraction time, selected cvi, selected algorithms, etc.
    additional_result_info = {"dataset": dataset_name}
    # Retrieve evaluated configurations from mkr
    EC = pd.read_csv(mkr_path / LearningPhase.evaluated_configs_filename, index_col=0)
    datasets_mkr = EC["dataset"].unique()

    ### (A1) Find similar dataset ###
    t0 = time.time()
    names, MF = extract_meta_features(X, mf_set)

    # Track runtime of meta-feature extraction
    mf_time = time.time() - t0
    additional_result_info["mf time"] = mf_time

    # Retrieve similar dataset
    D_s = find_similar_dataset(mkr_path, MF, dataset_name, mf_set)
    additional_result_info["similar dataset"] = D_s

    print(f"Most similar dataset is: {D_s}")
    # Retrieve evaluated configurations with ARI values for D_s
    ARI_s = retrieve_ARI_values_for_similar_dataset(EC, D_s)

    ### (A2) Select Cluster Validity Index ###
    if cvi == "predict":
        # Get Classification Model from MKR and use meta-features to predict a CVI
        cvi = predict_cvi(MF, dataset_name=dataset_name, mkr_path=mkr_path, mf_set=mf_set)
    additional_result_info["CVI"] = cvi.get_abbrev()
    print(f"Selected CVI: {cvi.name} ({cvi.get_abbrev()})")

    ### (A3) Select Warmstart Configurations ###
    warmstart_configs = select_warmstart_configurations(ARI_s, n_warmstarts)
    print(f"Selected Warmstart Configs:")
    print(warmstart_configs)

    ### (A4) Definition of Configurations Space (dependent on warmstart configurations) ###
    cs, warmstart_configs, algorithms = define_config_space(warmstart_configs, limit_cs)
    additional_result_info["algorithms"] = algorithms
    print(f"Selected Algorithms: {algorithms}")

    ### (A5) EnsOptimizer Loop ###
    opt_instance = optimizer(dataset=X,
                             true_labels=None,  # We do not have access to them in the application phase
                             cvi=cvi,
                             n_loops=n_optimizer_loops,
                             cs=cs,
                             wallclock_limit=time_limit
                             )
    opt_instance.optimize(initial_configs=warmstart_configs)

    print(f"Best obtained configuration is: {opt_instance.get_incumbent()}")
    return opt_instance, additional_result_info


if __name__ == '__main__':
    warnings.filterwarnings(category=RuntimeWarning, action="ignore")
    warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")

    # define random seed
    np.random.seed(1234)
    from sklearn.datasets import make_blobs

    X, y = make_blobs()

    run_application_phase(X=X, n_warmstarts=5, n_optimizer_loops=10)
