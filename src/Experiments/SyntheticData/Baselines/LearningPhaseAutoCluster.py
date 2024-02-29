from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Utils.Utils import get_type_from_dataset, get_noise_from_dataset, get_n_from_dataset
from Experiments.SyntheticData import DataGeneration
from MetaLearning import MetaFeatureExtractor
from MetaLearning.LearningPhase import run_learning_phase

datasets = DataGeneration.generate_datasets(n_values=[1000,  # 5000,
                                                      10000, 50000,
                                                      # 70000 --> Todo: Have to extract meta-features for them
                                                      ])

dataset_names = [name for name, data in datasets.items()]
dataset_names = sorted(dataset_names, key=lambda x: int(get_n_from_dataset(x)), #reverse=True
                       )
print(dataset_names)
dataset_df = pd.DataFrame(dataset_names, columns=["dataset"])
dataset_df["type"] = dataset_df["dataset"].apply(get_type_from_dataset)
dataset_df["noise"] = dataset_df["dataset"].apply(get_noise_from_dataset)
dataset_df = dataset_df.sort_values("dataset")
training_datasets_df, test_datasets_df = train_test_split(dataset_df, stratify=dataset_df[["type", "noise"]],
                                                          train_size=0.7, random_state=1234)
df = pd.read_csv(
    f"/home/ubuntu/automated_consensus_clustering/automated_consensus/src/EnsMetaLearning/synthetic_cf_es_m_pred_run0.csv")
test_datasets = df["dataset"].unique()

#print(test_datasets)
training_data_names = [training_dataset_name for training_dataset_name in
                       dataset_names if training_dataset_name not in test_datasets]

training_datasets = [datasets[training_data][0] for training_data in training_data_names]
training_labels = [datasets[training_data][1] for training_data in training_data_names]

for x, y, z in zip(training_data_names, [x.shape for x in training_datasets],
                   [len(np.unique(y)) for y in training_labels]):
    print(x)
    print(y)
    print(z)
    print("-------------")

for data in training_data_names:
    assert data not in test_datasets, f"{data} should not be in {test_datasets}"

meta_feature_path = Path("./")
mkr_path = Path("./")

for mf_set in [  #MetaFeatureExtractor.autocluster_mfs[0],
    MetaFeatureExtractor.landmarking_mfs[0]]:
    print(training_data_names)
    run_learning_phase(training_datasets, training_labels, training_data_names,
                       n_loops=100,
                       time_limit=180 * 60,
                       mf_path=meta_feature_path,
                       mf_set=mf_set,
                       skip_mf_extraction=False)
