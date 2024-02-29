from numpy import ndarray


def choose_best_automl(Y: ndarray, k_out: int) -> ndarray:
    """
    Returns best clustering when invoked on A4C-ensemble.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_out (int): desired number of clusters in consenus clustering
    Returns:
        ndarray of shape (N,): first column of Y, because A4C orders Y by 'goodness' this is the best clustering
    """
    return Y[:, 0]
