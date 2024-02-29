import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ConsensusCS.ConsensusCS import build_consensus_cs, CC_FUNCTIONS, CC_function_mapping
from EnsMetaLearning.old.Ensemble_CF_Prediction import train_cf_classifier, _prepare_k_ensemble_training_data, \
    generate_ensemble
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from Experiments.SyntheticData import DataGeneration
from Utils.Utils import get_type_from_dataset, get_noise_from_dataset, process_result_to_dataframe, \
    clean_up_optimizer_directory
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection

# Generate data
datasets = DataGeneration.generate_datasets(n_values=[1000,  # 5000,
                                                      10000, 50000,
                                                      # 70000 --> Todo: Have to extract meta-features for them
                                                      ])

X = [data[0] for _, data in datasets.items()]
y_labels = [data[1] for _, data in datasets.items()]

dataset_names = [d_name for d_name, _ in datasets.items()]
dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)
dataset_df = dataset_df.sort_values("dataset")
### Train-test split!
training_datasets_df, test_datasets_df = train_test_split(dataset_df, stratify=dataset_df[["type", "noise"]],
                                                          train_size=0.7, random_state=1234)

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

# Get Only Best Results for each dataset
# best_eval_ensemble = eval_ens.loc[eval_ens.groupby(["dataset"])["ARI"].idxmin()][
#    ["dataset", "ARI", "cc_function", "m", "es_selection", "type"]]
best_eval_ensemble = pd.DataFrame()
for dataset in eval_ens["dataset"].unique():
    dataset_eval_ens = eval_ens[eval_ens["dataset"] == dataset]
    print(dataset)
    # print(dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()])
    best_dataset_ari = dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()]
    if ("moons" in dataset or "circles" in dataset) and "MCLA" in best_dataset_ari["cc_function"]:
        print(best_dataset_ari[best_dataset_ari["cc_function"] == "MCLA"])
        best_eval_ensemble = best_eval_ensemble.append(best_dataset_ari[best_dataset_ari["cc_function"] == "MCLA"])
    else:
        print(best_dataset_ari[best_dataset_ari["runtime"] == best_dataset_ari["runtime"].min()])
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
CF_MODEL_FILE = 'model/cf_model.joblib'
cf_model = train_cf_classifier(X_metafeatures_train, y_cf_train)
dump(cf_model, CF_MODEL_FILE)
##############################################################
df_mfs_train = pd.DataFrame(X_metafeatures_train)
cc_function_order = ["ABV", "ACV", "MCLA", "QMI", "MM"]

# 2) Train k Prediction Model
K_PREDICTION_MODEL_FILE = "model/k_pred_ens.joblib"

df_mfs_cf_train, multi_label_targets = _prepare_k_ensemble_training_data(df_mfs_train, best_eval_ens_training)
print(df_mfs_cf_train)

ensemble_k_model = RandomForestClassifier(n_estimators=1000, random_state=1234,
                                          # min_samples_leaf=10 --> Leads to m= 0 in any case
                                          )
ensemble_k_model.fit(df_mfs_cf_train.to_numpy(), multi_label_targets)
dump(ensemble_k_model, K_PREDICTION_MODEL_FILE)

df_mfs_train, multi_label_targets = _prepare_k_ensemble_training_data(df_mfs_train, best_eval_ens_training,
                                                                      append_cfs=False)
print(df_mfs_train)

df_mfs_train = df_mfs_train.drop("tmp", axis=1)
ensemble_k_model_no_cf = RandomForestClassifier(n_estimators=1000, random_state=1234,
                                                # min_samples_leaf=10 --> Leads to m= 0 in any case
                                                )

ensemble_k_model_no_cf.fit(df_mfs_train.to_numpy(), multi_label_targets)


# dump(ensemble_k_model, K_PREDICTION_MODEL_FILE)

def run_optimization_all_three(X, y, ensemble, additional_result_info, mf_new=None):
    consensus_cs = build_consensus_cs(k_range=k_range)
    ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                      cs_generation=None,
                                      cs_consensus=consensus_cs,
                                      seed=1234)
    ens_optimizer.ensemble = ensemble
    consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=k_range)
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)
    consensus_result_df["Method"] = "Optimize All (CF x m x k)"

    print(consensus_result_df)
    clean_up_optimizer_directory(consensus_opt)
    return consensus_result_df


def run_each_cf_separately(X, y, ensemble, additional_result_info, mf_new=None):
    detailed_df = pd.DataFrame()

    for cc_function in CC_FUNCTIONS:
        consensus_cs = build_consensus_cs(cc_functions=[cc_function], k_range=k_range)
        ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                          cs_generation=None,
                                          cs_consensus=consensus_cs,
                                          seed=1234)
        ens_optimizer.ensemble = ensemble
        consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=k_range)
        consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                          additional_result_info,
                                                          ground_truth_clustering=y)
        consensus_result_df["CC_function"] = cc_function.get_name()
        clean_up_optimizer_directory(consensus_opt)
        detailed_df = pd.concat([detailed_df, consensus_result_df])
    return detailed_df


def run_no_gen_model(X, y, ensemble, additional_result_info, mf_new):
    cf_pred = cf_model.predict(mf_new.reshape(1, -1))[0]
    print(f"Predicted CF: {cf_pred}")
    cc_function = CC_function_mapping[cf_pred]
    consensus_cs = build_consensus_cs(cc_functions=[cc_function], k_range=k_range)

    ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                      cs_generation=None,
                                      cs_consensus=consensus_cs,
                                      seed=1234)
    ens_optimizer.ensemble = ensemble
    consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=k_range)
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)
    # consensus_result_df["CC_function"] = cc_function.get_name()
    clean_up_optimizer_directory(consensus_opt)
    return consensus_result_df


def run_no_cf_model(X, y, ensemble, additional_result_info, mf_new):
    m = ensemble.shape[1]
    consensus_cs = build_consensus_cs(cc_functions=CC_FUNCTIONS,
                                      k_range=k_range, default_ensemble_size=m,
                                      max_ensemble_size=None, step_size=None)

    ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                      cs_generation=None,
                                      cs_consensus=consensus_cs,
                                      seed=1234)
    ens_optimizer.ensemble = ensemble
    consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=k_range)
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)

    return consensus_result_df


def run_generation_model_and_cf_model(X, y, ensemble, additional_result_info, mf_new):
    best_gen_k = len(np.unique(ensemble[:, 0]))
    print(f"best k from generation: {best_gen_k}")
    consensus_cs = build_consensus_cs(cc_functions=CC_FUNCTIONS,
                                      k_range=(best_gen_k, best_gen_k))

    ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                      cs_generation=None,
                                      cs_consensus=consensus_cs,
                                      seed=1234)
    ens_optimizer.ensemble = ensemble
    consensus_opt = ens_optimizer.optimize_consensus(n_loops=no_optimizer_loops, k_range=k_range)
    consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                      additional_result_info,
                                                      ground_truth_clustering=y)
    return consensus_result_df


no_optimizer_loops = 40

if __name__ == '__main__':

    ##############################################################
    ##############################################################
    ##### Application Phase
    k_range = (2, 100)
    n_loops = 70

    approaches = {
        "No optimizer (k from generation)": run_generation_model_and_cf_model,
        "No Generation & No CF Model": run_each_cf_separately,  # Optimize m x k + run each CF separately
        "Optimize All (CF x m x k)": run_optimization_all_three,  # Optimize CF x m x k
        "No Generation Model": run_no_gen_model,
        "No CF Model: Optimize CFxk": run_no_cf_model,  # Generation Model + optimize CF x k
    }

    result_path = Path("optimization_study_results")
    result_file = "optimization_study.csv"
    if os.path.isfile(result_file):
        results = pd.read_csv(result_file)
    else:
        results = pd.DataFrame()

    print("test datasets:")
    print(test_datasets_df["dataset"].unique())

    for approach, approach_function in approaches.items():
        for test_dataset in test_datasets_df["dataset"].values:
            mf_new = mfs_df[mfs_df["dataset"] == test_dataset]
            mf_new = mf_new.drop("dataset", axis=1).to_numpy()
            mf_new = mf_scaler.transform(mf_new)

            dataset_index = dataset_names.index(test_dataset)
            X_test = X[dataset_index]
            y_test = y_labels[dataset_index]
            d_name = dataset_names[dataset_index]

            if "gaussian" in test_dataset or "varied" in test_dataset:
                cvi = CVICollection.CALINSKI_HARABASZ
            else:
                cvi = CVICollection.COP_SCORE

            print("-------------------------")
            print(f"Running approach {approach} on dataset {test_dataset}")

            # a bit hacky --> Actually, I should define the generation for each method separately
            if "No Generation" in approach or "Optimize All" in approach:
                ens, gen_info, cvi_scores = generate_ensemble(ensemble_k_values=list(range(k_range[0], k_range[1])),
                                                              X_new=X_test,
                                                              cvi=cvi, y_new=y_test)
            else:
                # Ensemble Generation using our model
                if "No CF Model" in approach or "No optimizer" in approach:
                    pred_k_values = ensemble_k_model_no_cf.predict(mf_new)[0]
                else:
                    pred_k_values = ensemble_k_model.predict(mf_new)[0]
                k_values_ens = [k + k_range[0] for k, value in enumerate(pred_k_values) if value == 1]
                ens, gen_info, cvi_scores = generate_ensemble(ensemble_k_values=k_values_ens,
                                                              X_new=X_test,
                                                              y_new=y_test, cvi=cvi)

            gen_info["cvi"] = cvi.get_abbrev()
            gen_info["dataset"] = d_name

            approach_result_path = result_path / approach
            approach_result_path.mkdir(exist_ok=True, parents=True)
            approach_result_df = approach_function(X_test, y_test, ens, gen_info, mf_new)
            approach_result_df["Method"] = str(approach)
            approach_result_df.to_csv(approach_result_path / (d_name + ".csv"), index=False)
