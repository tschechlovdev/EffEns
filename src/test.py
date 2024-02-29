from sklearn.datasets import make_blobs
from EnsMetaLearning.EffEns import EffEns
from automlclustering.ClusterValidityIndices import CVIHandler
from Utils.Utils import process_result_to_dataframe

# Generate simple synthetic data
X, y = make_blobs()

# Instantiate EffEnse. Use provided path to MKR.
effens = EffEns(path_to_mkr="./EffEnsMKR/")

# Choose CVI to evaluate results
cvi = CVIHandler.CVICollection.CALINSKI_HARABASZ
# Apply EffEns on Data X
result, _ = effens.apply_ensemble_clustering(X, cvi=cvi, n_loops=5)

# Parse Result
result = process_result_to_dataframe(result, {"cvi": cvi.get_abbrev()},
                                     # compare against ground-truth clustering
                                     ground_truth_clustering=y
                                     )

print(result[["iteration", "config", "CVI score", "Best NMI"]])
