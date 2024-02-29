import re
from pathlib import Path

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
import numpy as np
import os
import shutil

path_to_eval_results = Path("../../../eval_results")
path_to_eval_results.mkdir(exist_ok=True, parents=True)
print(str(path_to_eval_results))
def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


def get_noise_from_dataset(x):
    return x.split("-")[4].split("=")[1]


def get_type_from_dataset(x):
    return x.split("-")[0].split("=")[1]


def get_n_from_dataset(x):
    return x.split("-")[2].split("=")[1]


def compute_nmi_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


def clean_up_optimizer_directory(optimizer_instance):
    if os.path.exists(optimizer_instance.output_dir) and os.path.isdir(optimizer_instance.output_dir):
        shutil.rmtree(optimizer_instance.output_dir)


def process_result_to_dataframe(optimizer_result, additional_info, ground_truth_clustering):
    selected_cvi = additional_info["cvi"]
    # The result of the application phase an optimizer instance that holds the history of executed
    # configurations with their runtime, cvi score, and so on.
    # We can also access the predicted clustering labels of each configuration to compute ARI.
    optimizer_result_df = optimizer_result.get_runhistory_df()

    for key, value in additional_info.items():
        # if key == "algorithms":
        #     value = "+".join(value)
        if key == "similar dataset":
            value = "+".join(value)
        print(key)
        print(value)

        optimizer_result_df[key] = value

    # optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
    # optimizer_result_df["iteration"] = [i + 1 for i in range(len(optimizer_result_df))]

    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi]
    optimizer_result_df = optimizer_result_df.sort_values("iteration")
    optimizer_result_df["wallclock time"] = optimizer_result_df["runtime"].cumsum()

    optimizer_result_df['Best CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, ground_truth_clustering)
    optimizer_result_df['NMI'] = compute_nmi_values(optimizer_result_df, ground_truth_clustering)

    optimizer_result_df['Best ARI'] = optimizer_result_df.apply(
        lambda row:
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"] == row['Best CVI score'])]["ARI"].values[0],
        axis=1)
    optimizer_result_df['Best NMI'] = optimizer_result_df.apply(
        lambda row:
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"] == row['Best CVI score'])]["NMI"].values[0],
        axis=1)

    best_row = optimizer_result_df.iloc[optimizer_result_df['CVI score'].idxmin()]
    optimizer_result_df["best config"] = str(optimizer_result.get_incumbent().get_dictionary())
    optimizer_result_df["k_pred"] = [len(np.unique(best_row["labels"])) for _ in range(len(optimizer_result_df))]
    optimizer_result_df["config_ranking"] = optimizer_result_df["CVI score"].rank()

    # We do not need the labels in the CSV file
    # optimizer_result_df = optimizer_result_df.drop("labels", axis=1)
    optimizer_result_df = optimizer_result_df.drop(selected_cvi, axis=1)
    optimizer_result_df = optimizer_result_df.drop("labels", axis=1)
    print(optimizer_result_df)

    return optimizer_result_df


def get_n_from_real_world_data(d_name):
    return int(re.search("_n\d\d\d\d?\d?_", str(d_name)).group().replace("_n", "").replace("_", ""))


def calculate_gen_info(X_new, y_new, effEns, cvi):
    ensemble = effEns.get_ensemble()
    assert ensemble.shape[0] == X_new.shape[0]
    cvi_gen_scores = [cvi.score_cvi(X_new, ensemble[:, i]) for i, _ in
                      enumerate(list(range(ensemble.shape[1])))]
    # Parse best single results from ensemble generation and CVI scores
    best_gen_idx = np.argmin(cvi_gen_scores)
    best_gen_nmi = CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, true_labels=y_new,
                                                           labels=ensemble[:, best_gen_idx])
    best_gen_ari = CVICollection.ADJUSTED_RAND.score_cvi(data=None, true_labels=y_new,
                                                         labels=ensemble[:, best_gen_idx])
    best_gen_cvi = cvi_gen_scores[best_gen_idx]
    return {"gen_ari": best_gen_ari, "gen_nmi": best_gen_nmi, "gen_cvi": best_gen_cvi,
            "m": ensemble.shape[1]}
