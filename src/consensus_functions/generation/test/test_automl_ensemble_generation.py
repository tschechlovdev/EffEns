import unittest

import numpy as np
from sklearn.datasets import make_blobs

from ens_clust.generation.automl_ensemble_generation import automl_ensemble


class AutoMLEnsembleGenerationTest(unittest.TestCase):

    def test_automl_ensemble_array_creation(self):
        # Here, we create a simple artificial dataset with sklearn
        # default number of clusters is here 3 (but can be configured in make_blobs)
        N = 10000
        X, y = make_blobs(n_samples=N)
        nr_partitions = 10
        # Nxnr_partitions array containing clustering results/partitions
        ensemble, run_history = automl_ensemble(X, nr_partitions, return_run_history=True)
        # if nr_configurations is not passed to automl_ensemble it defautls to nr_partitions
        nr_configurations = nr_partitions
        self.assertEqual(len(run_history), nr_configurations)
        self.assertEqual(ensemble.shape, (N, nr_partitions))
        np.testing.assert_array_equal(ensemble[:, 0], run_history[0].labels)

    def test_nr_configurations(self):
        # Here, we create a simple artificial dataset with sklearn
        # default number of clusters is here 3 (but can be configured in make_blobs)
        N = 10000
        X, y = make_blobs(n_samples=N)
        nr_partitions = 10
        # Define number of loops, i.e., number of configurations to execute.
        nr_configurations = 3 * nr_partitions
        # Nxnr_partitions array containing clustering results/partitions
        ensemble, run_history = automl_ensemble(X, nr_partitions, nr_configurations, return_run_history=True)
        self.assertEqual(len(run_history), nr_configurations)
        self.assertEqual(ensemble.shape, (N, nr_partitions))
        np.testing.assert_array_equal(ensemble[:, 0], run_history[0].labels)


if __name__ == '__main__':
    unittest.main()
