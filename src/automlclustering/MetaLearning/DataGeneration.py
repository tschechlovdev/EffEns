import time
from pathlib import Path

from sklearn.cluster import DBSCAN, SpectralClustering, AffinityPropagation, MeanShift, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.utils import check_random_state

from Metrics.MetricHandler import MetricCollection

random_state = 1234
np.random.seed(random_state)

DATASET_TYPES = ['gaussian','circles', 'moons', 'varied']

def generate_datasets(dataset_types=DATASET_TYPES):
    different_shape_sets = {}
    # Definition of dataset characteristics --> Define number of instances, attributes, #clusters
    characteristics = {'k': [10, 30, 50],
                       'n': [
                             1000, 5000, 10000
                             #100, 500, 1000
                             ],
                       'd': [10, 30, 50],
                       'noise': [
                           0.0, 0.01,
                           0.05, 0.1],
                       'type': dataset_types}

    for n in characteristics['n']:
        for data_type in characteristics['type']:
            generator = check_random_state(random_state)

            if data_type == 'gaussian':
                noise = 0
                # gaussian and varied also have "k" value
                for k in characteristics['k']:
                    for d in characteristics['d']:
                        data = make_blobs(n_samples=n, n_features=d, centers=k, random_state=random_state)
                        if noise > 0:
                            # add sklearn methodology of adding noise --> Adds to EACH point a standard deviation
                            # given by noise.
                            data[0] += generator.normal(scale=noise, size=data[0].shape)
                        different_shape_sets[f"type={data_type}-k={k}-n={n}-d={d}-noise={noise}"] = data
            elif data_type == 'varied':
                noise = 0
                for k in characteristics['k']:
                    for d in characteristics['d']:
                        data = make_blobs(n_samples=n,
                                          n_features=d,
                                          centers=k,
                                          # varying cluster std for each cluster --> highly affected by "K"!
                                          cluster_std=[0.5 + i / k for i in list(range(1, k + 1))],
                                          random_state=random_state)
                        if noise > 0:
                            # add sklearn methodology of adding noise --> Adds to EACH point a standard deviation
                            # given by noise.
                            data[0] += generator.normal(scale=noise, size=data[0].shape)

                        different_shape_sets[f"type={data_type}-k={k}-n={n}-d={d}-noise={noise}"] = data

            elif data_type == 'circles' or data_type == 'moons':
                k = 2
                d = 2
                for noise in characteristics['noise']:
                    if data_type == 'circles':
                        data = make_circles(n_samples=n, factor=0.5, noise=noise, random_state=random_state)
                    elif data_type == 'moons':
                        data = make_moons(n_samples=n, noise=noise, random_state=random_state)

                    different_shape_sets[f"type={data_type}-k={k}-n={n}-d={d}-noise={noise}"] = data
    return different_shape_sets

if __name__ == '__main__':
    datasets = generate_datasets()
    print(len(datasets))
    print(datasets.keys())
    import pandas as pd
    related_work_offline_result = pd.read_csv(Path('/volume/related_work') / 'related_work_offline_opt.csv', index_col=None)
    print(related_work_offline_result.isna().sum())
    related_work_offline_result.dropna(inplace=True)
    print(related_work_offline_result.isna().sum())

    # get the training data for mlp
    X = related_work_offline_result[[metric.get_abbrev() for metric in MetricCollection.internal_metrics]].to_numpy()
    y = related_work_offline_result['ARI'].to_numpy()

    t0 = time.time()
    # train the mlp
    mlp = MLPRegressor(hidden_layer_sizes=(60, 30, 10), activation='relu')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mlp.fit(X, y)
    y_pred = mlp.predict(X_test)
    print(y_pred)
    print(y_test)
    print(y_test - y_pred)
    print(f"Training MLP took {time.time() - t0}s")

    print(mlp.predict(X[10].reshape(1,-1)))
    print(y[10])

    print(f"Score is: {r2_score(y_pred, y_test)}")

    data = make_blobs(n_samples=10000, n_features=100, centers=100, random_state=random_state)
    X = data[0]
    t0 = time.time()
    y_pred = KMeans().fit_predict(X)
    t0 = time.time()
    MetricCollection.DENSITY_BASED_VALIDATION.score_metric(X, y_pred)
    print(f"took {time.time() - t0}s")
