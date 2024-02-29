import time

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

from Utils.RAMManager import memory


class Cas:

    def __init__(self, base_labels=None) -> None:
        if base_labels is not None:
            self.nmi_matrix = self._compute_nmi_matrix(base_labels)
        else:
            self.nmi_matrix = None

    @memory(percentage=1.5)
    def cluster_and_select(self, base_labels: np.ndarray, m: int):
        """
        Get m clusters out of the base_labels based on the cas method
        
        ### Parameters
        
        base_labels: the labels from which to select the *m* most appropriate ones
        m: the number of result solutions (must be >=1 and <= len(base_labels))
        """

        assert m <= len(base_labels), 'm cannot be greater than the number of available clusters'
        assert m >= 0, 'm cannot be smaller than 1'

        # Special case, m == number of base clusters, in this case all clusters are returned
        if m == len(base_labels): return base_labels
        print(base_labels.shape)

        if self.nmi_matrix is None:
            self.nmi_matrix = self._compute_nmi_matrix(base_labels)

        print("Running Spectral Clustering")
        nmi_clustering_result = SpectralClustering(n_clusters=m,
                                                   affinity="precomputed",
                                                   random_state=42).fit_predict(self.nmi_matrix)
        print("finished")

        n = base_labels.shape[0]
        selected_ensemble = np.zeros((n, m))

        for cluster in np.unique(nmi_clustering_result):
            # get indices from nmi_clustering_result
            cluster_indices = np.where(cluster == nmi_clustering_result)[0]
            # Get this specific cluster
            nmi_cluster = self.nmi_matrix[cluster_indices]
            # Calculate mean of NMI values for this cluster
            nmi_mean_instances = np.mean(nmi_cluster, axis=1)
            # Get instance with maximum (average) NMI value
            max_instance = np.argmax(nmi_mean_instances)
            # Get the clustering result with maximum (average) NMI value
            ens_result_to_use = cluster_indices[max_instance]
            selected_ensemble[:, cluster] = base_labels[:, ens_result_to_use]

        return selected_ensemble

    @memory(percentage=1.5)
    def _compute_nmi_matrix(self, base_labels):
        print("Creating NMI matrix")
        nmi_start = time.time()
        nmi_matrix = np.zeros((base_labels.shape[1], base_labels.shape[1]))
        for i in range(base_labels.shape[1]):
            for j in range(i, base_labels.shape[1]):
                nmi_matrix[i, j] = normalized_mutual_info_score(base_labels[:, i], base_labels[:, j])
        print(f"Finished NMI Matrix, took {time.time() - nmi_start}s")

        return nmi_matrix


import numpy as np


def main():
    label1 = np.array([1, 1, 1, 2, 2, 3, 3])
    label2 = np.array([2, 2, 2, 3, 3, 1, 1])
    label3 = np.array([4, 4, 2, 2, 3, 3, 3])
    label4 = np.array([1, 2, 3, 1, 2, 3, 3])
    label5 = np.array([4, 2, 3, 4, 2, 3, 4])

    base_labels = np.array([label1, label2, label3, label4, label5])

    m = 2

    cas = Cas()

    result = cas.cluster_and_select(base_labels=base_labels, m=m)

    print(result)


if __name__ == '__main__':
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    n = 500
    X, y = make_blobs(n_samples=n)

    k_range = (2, 100)
    m = 50

    ens = np.zeros((X.shape[0], len(list(range(k_range[0], k_range[1] + 1)))))
    for k in range(k_range[0], k_range[1] + 1):
        print(f"Running k-Means with k={k}")
        y_pred = KMeans(n_clusters=k, n_init=1).fit_predict(X)
        ens[:, k - k_range[0]] = y_pred
        print(f"Finished k-Means")

    print(f"Running CAS with k={m}")
    start_cas = time.time()
    ens = Cas(ens).cluster_and_select(ens, m)
    print(f"Finished CAS, took {time.time() - start_cas}s")
    # print(ens.transpose())
    # print(Cas().cluster_and_select(ens.transpose(), 5))
    for i in range(ens.shape[1]):
        print(len(np.unique(ens[:, i])))
