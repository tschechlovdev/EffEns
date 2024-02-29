import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from sklearn.cluster import OPTICS

from Utils.Utils import get_n_from_real_world_data

dataset_file_names = [file for file in os.listdir("/volume/datasets/real_world") if ".csv" in file]

dataset_file_names = sorted(dataset_file_names, key=get_n_from_real_world_data  # , reverse=True
                            )
summary_path = Path(f"results/Baselines/")

result_path = summary_path / "OPTICS"
result_path.mkdir(parents=True, exist_ok=True)

optics_result = pd.DataFrame()
for data_file_name in dataset_file_names:
    df = pd.read_csv(f"/volume/datasets/real_world/{data_file_name}",
                     index_col=None, header=None)
    X = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]

    print("----------------")
    print(f"Running method OPTICS on dataset {data_file_name}")
    start = time.time()
    y_pred = OPTICS().fit_predict(X)
    runtime = time.time() - start
    print(f"Finished OPTICS on {data_file_name} in {runtime}s")

    print("Calculating NMI score")
    ami_score = CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, labels=y_pred, true_labels=y_test)
    print(f"Finished NMI score with score {ami_score}")

    print("Calculating ARI score")
    ari_score = CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=y_pred, true_labels=y_test)
    print(f"Finished ARI score with score {ari_score}")

    true_k = len(np.unique(y_test))
    n = X.shape[0]

    f = X.shape[1]
    #noise = float(data_file_name.replace(".csv", "").split("-")[4].split("=")[-1])
    type_ = data_file_name.split("-")[0].split("=")[-1]

    result_info = {"dataset": [data_file_name],
                   "cvi": [None],
                   "n": [n],
                   "f": [f],
                   "true_k": [true_k],
                   #"noise": [noise],
                   "type": [type_],
                   #"run": 0,
                   "Method": ["OPTICS"],
                   "runtime": [runtime],
                   "Best ARI": [ari_score],
                   "Best NMI": [ami_score]
                   }

    optics_result = pd.concat([optics_result, pd.DataFrame(result_info)])
    optics_result.to_csv(result_path / "optics_result.csv",  index=False)