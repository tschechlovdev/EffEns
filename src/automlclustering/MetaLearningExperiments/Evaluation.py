import warnings
import pandas as pd
from pathlib import Path

from pandas.errors import SettingWithCopyWarning

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.MetaLearningExperiments import DataGeneration
from automlclustering.MetaLearning import LearningPhase
from automlclustering.MetaLearning.ApplicationPhase import run_application_phase
from automlclustering.Helper.Helper import _add_iteration_metric_wallclock_time, _add_missing_iterations

warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")
warnings.filterwarnings(category=UserWarning, action="ignore")


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


if __name__ == '__main__':
    shape_sets = DataGeneration.generate_datasets()
    datasets = [dataset[0] for key, dataset in shape_sets.items()]
    dataset_names = list(shape_sets.keys())
    true_labels = [dataset[1] for key, dataset in shape_sets.items()]
    mkr_path = LearningPhase.mkr_path

    # Parameters for our experiments
    n_warmstarts = 25
    n_loops = 100
    limit_cs = True
    time_limit = 120 * 60
    cvi = "predict"
    mf_set = ["statistical", "info-theory", "general"]

    path_to_store_results = Path("./evaluation_results")
    if not path_to_store_results.exists():
        path_to_store_results.mkdir(exist_ok=True, parents=True)

    evaluation_results = pd.DataFrame()

    for dataset, ground_truth_labels, dataset_name in zip(datasets, true_labels, dataset_names):
        # Run the application phase for each "new" dataset.
        # Note that we have executed the learning phase for the new dataset as well, however we skip the meta-knowledge
        # for it. For this, we use the dataset_name.
        optimizer_instance, additional_result_info = run_application_phase(X=dataset, mkr_path=mkr_path,
                                                                           dataset_name=dataset_name,
                                                                           n_warmstarts=n_warmstarts,
                                                                           n_optimizer_loops=n_loops, cvi=cvi,
                                                                           limit_cs=limit_cs,
                                                                           time_limit=time_limit,
                                                                           mf_set=mf_set)
        selected_cvi = additional_result_info["CVI"]
        # The result of the application phase an optimizer instance that holds the history of executed configurations
        # with their runtime, cvi score, and so on.
        # We can also access the predicted clustering labels of each configuration to compute ARI.
        optimizer_result_df = optimizer_instance.get_runhistory_df()
        for key, value in additional_result_info.items():
            if isinstance(value, list):
                value = "+".join(value)
            optimizer_result_df[key] = value

        # Preprocess results, add some columns, prune to only have the best configurations over time
        optimizer_result_df = _add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
        print(optimizer_result_df)

        # Compute ARI values for the evaluation as we have ground-truth clusterings
        optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, ground_truth_labels)

        # As we have pruned for the best CVI results, we copy the iterations that did not improve with the best CVI
        # value. This way we can easily plot the data afterwards.
        optimizer_result_df = _add_missing_iterations(optimizer_result_df, n_loops)
        evaluation_results = pd.concat([evaluation_results, optimizer_result_df])

        #
        evaluation_results.to_csv(Path(path_to_store_results) / f"results_{mf_set}.csv")
