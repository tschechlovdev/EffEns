# EffEns: Efficient Ensemble Clustering based on Meta-Learning and Hyperparameter Optimization 

Prototypical Implementation in Python of the submitted Paper "Ensemble  Clustering based on Meta-Learning and Hyperparameter Optimization" at VLDB 2024.
In the following, we provide an overview of the code structure, an installation instruction, and an example on how to use EffEns.

## Overview

The main code is in the "src" folder. It contains the following modules:

- ``automlclustering``: Contains the adapted code from [AutoML4Clust](https://github.com/tschechlovdev/Automl4Clust) and [ML2DAC](https://github.com/tschechlovdev/ml2dac/tree/main), which provide 
    implementations of AutoML for Clustering Systems and for different meta-feature sets.
- ``consensus_functions``: Contains implementations of the five consensus functions "ABV", "ACV", "MCLA", "MM", and "QMI", which we used in our paper.
- ``ConsensusCS``: Provides the consensus functions and hyperparameters as configuration space for the optimizer.
- ``EffEnsMKR``: Contains a script that stores the path to the MKR and the filenames of the "evaluated ensembles" and the meta-features.
- ``EnsMetaLearning``: All functionality for our meta-learning procedure. 
In particular, for the learning phase to evaluate different ensemble subsets and extract the meta-features.
    It also contains ``EffEns`` that can be applied on new datasets.
- ``EnsOptimizer``: Contains the optimizer that we use for hyperparameter optimization of the consensus functions. 
  We use SMAC as optimizer and provide a wrapper class as well as the black box function for optimization.
- ``Experiments``: Contains the code for the experiments for the synthetic and real-world datasets of our paper (cf. Section 7).
- ``Utils``: Contains some utility code such as functions to process the optimizer results or to clean up temporary directories.

Note that the directory ``real_world_datasets`` contains the datasets that we used in our experiments.
Further, ``evaluation_results`` contains all results from our experimental evaluatioin. 

## Installation

Our implementation is based on Python and we require Python 3.9.
Furthermore, as SMAC only runs on Linux, we also require a Linux system.
We have tested on Ubuntu 20.04.

Before installing EffEns, you first have to install the following that are required for some of the libraries:
- ``sudo apt-get install build-essential``
- ``sudo apt-get install gcc``

The easiest way of installing EffEns is to use Anaconda. Follow https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
to install Anaconda.
We will then create a prepared Python 3.9 environment:
- ``conda env create -f environment.yml``

This should create a conda environment with the name "automated_ensemble_clustering".
Then you have to install ib_base as it is not available as package: 

```git clone https://collaborating.tuhh.de/cip3725/ib_base.git
cd ib_base
python setup.py install
cd ..
```

After finishing this, you have to add the "src" folder of EffEns and the path to "ib_base" to your PYTHONPATH
You may also have to add them to your conda path
``gedit  ~/anaconda3/envs/automated_ensemble_clustering/lib/python3.9/site-packages/conda.pth``

Now everything should be setup and you can try to run ``python src/Experiments/SyntehticData/EffEns_Experiment_synthetic.py``.
This should run without any errors.

## Examples

In the following, we provide a simple example on how to use EffEns on new unseen datasets with the provided Meta-Knowledge Repository:

```Python
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
```
