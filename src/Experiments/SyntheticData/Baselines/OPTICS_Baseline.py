import time
from pathlib import Path

import numpy as np
import pandas as pd
from ClusterValidityIndices.CVIHandler import CVICollection
from sklearn.cluster import OPTICS
from sklearn.model_selection import train_test_split

from EnsMetaLearning.Helper import get_type_from_dataset, get_noise_from_dataset
from Experiments.SyntheticData import DataGeneration

datasets = DataGeneration.generate_datasets(n_values=[1000, 10000, 50000])

data_X = [data[0] for _, data in datasets.items()]
y_labels = [data[1] for _, data in datasets.items()]

dataset_names = [d_name for d_name, _ in datasets.items()]
dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)

### Train-test split!
training_datasets_df, test_datasets_df = train_test_split(dataset_df, stratify=dataset_df[["type", "noise"]],
                                                          train_size=0.7, random_state=1234)

print(test_datasets_df)
print(len(test_datasets_df))

## Parameters
test_data_indices = list(test_datasets_df.index)
print(test_data_indices)

optics_result = pd.DataFrame()
#result_path = Path(
#    f"/home/ubuntu/automated_consensus_clustering/automated_consensus/src/Experiments/SyntheticData/results/baselines/OPTICS")
result_path = Path(f"../results/baselines/OPTICS")
result_path.mkdir(parents=True, exist_ok=True)
for test_index in test_data_indices:
    dataset_name = dataset_names[test_index]
    X = data_X[test_index]
    y_test = y_labels[test_index]
    print("----------------")
    print(f"Running method OPTICS on dataset {dataset_name}")
    start = time.time()
    y_pred = OPTICS().fit_predict(X)
    runtime = time.time() - start
    print(f"Finished OPTICS on {dataset_name} in {runtime}s")

    print("Calculating NMI score")
    ami_score = CVICollection.ADJUSTED_MUTUAL.score_cvi(data=None, labels=y_pred, true_labels=y_test)
    print(f"Finished NMI score with score {ami_score}")

    print("Calculating ARI score")
    ari_score = CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=y_pred, true_labels=y_test)
    print(f"Finished ARI score with score {ari_score}")

    true_k = len(np.unique(y_test))
    n = X.shape[0]

    f = X.shape[1]
    noise = float(dataset_name.replace(".csv", "").split("-")[4].split("=")[-1])
    type_ = dataset_name.split("-")[0].split("=")[-1]

    result_info = {"dataset": [dataset_name],
                   "cvi": [None],
                   "n": [n],
                   "f": [f],
                   "true_k": [true_k],
                   "noise": [noise],
                   "type": [type_],
                   #"run": 0,
                   "Method": ["OPTICS"],
                   "runtime": [runtime],
                   "Best ARI": [ari_score],
                   "Best NMI": [ami_score]
                   }

    optics_result = pd.concat([optics_result, pd.DataFrame(result_info)])
    optics_result.to_csv(result_path / "optics_result.csv",  index=False)
