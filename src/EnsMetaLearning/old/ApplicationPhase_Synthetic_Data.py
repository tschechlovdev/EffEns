import os

os.sys.path.append("/home/ubuntu/automated_consensus_clustering/automated_consensus/src")

from Experiments.SyntheticData import DataGeneration

import pandas as pd

from ClusterValidityIndices import CVIHandler
from ApplicationPhase import run_ensemble
from MetaLearning import MetaFeatureExtractor

if __name__ == '__main__':
    datasets = DataGeneration.generate_datasets(n_values=[1000, 5000, 10000, 50000,
                                                          # 70000,
                                                          # 100000,
                                                          # 500000, 1000000
                                                          ])
    # mf_per_dataset = {d_name: MetaFeatureExtractor.extract_meta_features(data[0], mf_set=["statistical", "general"],)[1]
    #                     for d_name, data in datasets.items()}
    #
    # df = pd.DataFrame(data=mf_per_dataset)
    # print(df.T)
    # df = df.T
    # df["dataset"] = df.index
    # df.to_csv("meta_features.csv",  index=None)

    all_results = pd.DataFrame()

    result_file_name = "synthetic_eval_classif_cvi.csv"
    if os.path.isfile(result_file_name):
        all_results = pd.read_csv(result_file_name)
    else:
        all_results = pd.DataFrame()

    for d_name, data in datasets.items():
        X_new = data[0]
        y_new = data[1]

        if "gauss" in d_name or "varied" in d_name:
            cvis = [CVIHandler.CVICollection.CALINSKI_HARABASZ,
                    CVIHandler.CVICollection.DAVIES_BOULDIN]
        else:
            cvis = [CVIHandler.CVICollection.DENSITY_BASED_VALIDATION]

        for cvi in cvis:
            for use_classifier in [True, False]:
                for reduce_cs in [True, False]:
                    for run in range(3):
                        existing_results = all_results[(all_results["dataset"] == d_name)
                                                       & (all_results["cvi"] == cvi.get_abbrev())
                                                       & (all_results["reduce_cs"] == reduce_cs)
                                                       & (all_results["classifier"] == use_classifier)
                                                       & (all_results["run"] == run)]
                        if not len(existing_results) == 0:
                            # We already have this result, so continue
                            print(
                                f"Continue for {d_name}, {cvi.get_abbrev()}, clf={use_classifier}, reduce_cs={reduce_cs}, run={run}")
                            continue
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
                        all_results.to_csv(result_file_name, index=False)
