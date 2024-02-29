import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import check_random_state

# np.random.seed(random_state)

DATASET_TYPES = ['circles',
                 'moons',
                 'gaussian',
                 'varied']


def generate_datasets(dataset_types=DATASET_TYPES, n_values=None, n_feature_values=None,
                      k_values=None, random_state=2, noise_values=None):
    different_shape_sets = {}
    # Definition of dataset characteristics --> Define number of instances, attributes, #clusters
    characteristics = {'k': [10, 30, 50] if not k_values else k_values,
                       'n': [1000,
                             10000,
                             25000,
                             50000,
                             75000,
                             100000,
                             500000  # That might break?
                             ] if not n_values else n_values,
                       'd': [100, 50, 75] if not n_feature_values else n_feature_values,
                       'noise': [
                           0.0, 0.01,
                           0.05, 0.1] if not noise_values else noise_values,
                       'type': dataset_types}

    for n in characteristics['n']:
        for data_type in characteristics['type']:
            generator = check_random_state(random_state)

            if data_type == 'gaussian':
                # We could also use more noise here
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
                        cluster_std = [0.5, 0.75, 0.25] * 100
                        cluster_std = cluster_std[0:k]
                        data = make_blobs(n_samples=n,
                                          n_features=d,
                                          centers=k,
                                          # varying cluster std for each cluster
                                          # Note that this is affected by "k"
                                          cluster_std=cluster_std,
                                          random_state=random_state,
                                          center_box=[-1, 1]
                                          )
                        if noise > 0:
                            # add sklearn methodology of adding noise --> Adds to EACH point a standard deviation
                            # given by noise.
                            data[0] += generator.normal(scale=noise, size=data[0].shape, random_state=random_state)

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
    # Script to store the generated datasets
    data_mapping = generate_datasets(n_values=[500, 1000])
    for data_name, data in data_mapping.items():
        X = data[0]
        y = data[1]
        X_new = np.zeros((X.shape[0], X.shape[1] + 1))
        X_new[:, :-1] = X
        X_new[:, -1] = y
        print(X)
        print(y)
        print(X_new)
        print(adjusted_rand_score(X_new[:, -1], y))
        df = pd.DataFrame(X_new)
        df.to_csv(f"/volume/datasets/synthetic/{data_name}.csv", header=None, index=False)
        # pd.DataFrame(data=X)
