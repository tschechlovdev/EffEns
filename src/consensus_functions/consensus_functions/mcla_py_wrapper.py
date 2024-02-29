from consensus_functions.consensus_functions.mcla_gregory_giecold import MCLA
from numpy import ndarray


def mcla(Y: ndarray, k_max: int, verbose=False) -> ndarray:
    """
    Given the label matrix L of an ensemble with N points and H base clusterings,
    compute consensus_clustering with at most k_max clusters.
    A third party MCLA implementation is invoked.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_max (number): desired number of output clusters, the number of clusters returned could be less
        if some meta-cluster cannot claim a single point
    """
    return MCLA(cluster_runs=Y.T, verbose=verbose, N_clusters_max=k_max)
