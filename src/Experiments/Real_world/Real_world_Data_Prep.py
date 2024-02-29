# from sklearn.cluster import OPTICS
import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    seed = 2
    np.random.seed(seed)
    ## Parameters
    dataset_file_names = [file for file in os.listdir("/volume/datasets/synthetic") if ".csv" in file]

    real_world_datasets = ["miceprotein",
                           "har",
                           "pendigits",
                           "letter",
                           "USPS",
                           "mnist_784",
                           "Fashion-MNIST"]
    mnist_subsets = [10000, 20000, 30000, 40000, 50000, 60000, 70000]

    for data in real_world_datasets:
        print(f"Fetching data {data}...")
        X, y = fetch_openml(data, return_X_y=True)
        y = LabelEncoder().fit_transform(y)
        c = len(np.unique(y))
        if "mnist" in data or "MNIST" in data:
            subsets = mnist_subsets
        else:
            subsets = [X.shape[0]]

        for subset_size in subsets:
            if subset_size < X.shape[0]:
                X_train, X_test, \
                    y_train, y_test = train_test_split(X, y,
                                                       train_size=subset_size, stratify=y,
                                                       random_state=seed)
            else:
                X_train = X
                y_train = y

            from sklearn.cluster import KMeans
            from sklearn.impute import KNNImputer

            print(f"data has shape: {X_train.shape}")
            print(f"... and classes: {len(np.unique(y))}")

            # if np.isnan(X).any():
            X_train = KNNImputer(n_neighbors=10).fit_transform(X_train)

            print(f"Running kmeans...")
            start = time.time()
            km = KMeans(n_clusters=len(np.unique(y)))
            y_pred = km.fit_predict(X_train)
            ari = adjusted_rand_score(y_train, y_pred)
            print(f"Finished on data: {data}")
            print(f"ARI score: {ari}")
            print(f"Runtime: {time.time() - start}s")
            print("---------------------------------------")

            X_new = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
            X_new[:, :-1] = X_train
            X_new[:, -1] = y_train
            df = pd.DataFrame(X_new)
            df.to_csv(f"/volume/datasets/real_world/{data}_n{subset_size}_f{X_train.shape[1]}_c{c}.csv", header=False,
                      index=False)
