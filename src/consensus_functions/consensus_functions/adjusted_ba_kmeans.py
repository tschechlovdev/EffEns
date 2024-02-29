import time

import numpy as np
from sklearn.cluster import KMeans
from numpy import ndarray
from consensus_functions.datastructures.datastructures import adjusted_ba


def adjusted_ba_kmeans(Y: ndarray, k_out: int, n_init: int = 1, random_state=None, verbose=False):
    """
    Given the label matrix Y of an ensemble with N points and M base clusterings,
    compute consensus_clustering with k_out clusters.
    The consensus clustering is the result of K-means with k_out clusters applied to
    a normalized version of the binary association matrix of the ensemble.
    See pages 33-34 of BA.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_out (int): desired number of clusters in consensus clustering
        n_init (int): Number of time the K-means algorithm will be run with different centroid seeds on the adjusted BA. 
        The final results will be the best output of n_init consecutive runs in terms of SSE.
        random_state (number): Determines random number generation for centroid initialization in K-means.
        verbose (bool): iff true, return the adjusted binary association matrix
    Returns:
        ndarray of shape (N,): consensus clustering
    """
    print(f"QMI: Generating BA matrix")
    start = time.time()
    adjusted_ba_matrix = adjusted_ba(Y)
    print(f"Finished ba matrix, took: {time.time() - start}")
    adjusted_ba_matrix = np.asarray(adjusted_ba_matrix)
    print(f"QMI: Start Running KMeans on BA Matrix with shape {adjusted_ba_matrix.shape}")
    start = time.time()
    labels = KMeans(n_clusters=k_out,
                    n_init=n_init,
                    random_state=random_state,
                    max_iter=100).fit_predict(adjusted_ba_matrix)
    print(f"Finished KMeans on BA matrix, took: {time.time() - start}")

    return [labels] + [adjusted_ba_matrix] if verbose else labels


qmi = adjusted_ba_kmeans
