from kmodes.kmodes import KModes
from numpy import ndarray


def iterative_voting_consensus(L: ndarray, k_out: int, n_init: int = 1, n_jobs: int = 1,
                               max_iter=10):
    """
    Given the label matrix L of an ensemble with N points and M base clusterings,
    compute consensus_clustering with k_out clusters.
    The consensus clustering is the result of K-modes with k_out clusters applied to
    the rows of L (the ensemble). The rows have M attributes of categorical type.
    Each attribute corresponds to the cluster label assigned to the point in a partition of the ensemble.
    See page 24 and 33 of BA.
    Args:
        L (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_out (int): desired number of clusters in consenus clustering
        n_init (int): Number of times the K-modes algorithm runs.
        n_jobs (int): Number of concurrent runs of K-modes.
        max_iter (int): Number of iterations to run K-modes at maximum
    Returns:
        ndarray of shape (N,): consensus clustering is the best of n_init K-modes
        results in terms of sum of hamming distances
    """
    km = KModes(n_clusters=k_out, n_init=n_init, n_jobs=n_jobs, max_iter=max_iter, init="huang")
    return km.fit_predict(L)
