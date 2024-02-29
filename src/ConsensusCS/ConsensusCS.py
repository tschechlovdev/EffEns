from enum import Enum, auto
from typing import Union

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter, Constant

from Utils.RAMManager import memory
from consensus_functions.consensus_functions.adjusted_ba_kmeans import adjusted_ba_kmeans, qmi
from consensus_functions.consensus_functions.cvs import ada_bvote, ada_cvote
from consensus_functions.consensus_functions.ivc import iterative_voting_consensus
from consensus_functions.consensus_functions.mcla_py_wrapper import mcla
from consensus_functions.consensus_functions.mixture_model import mm


class BaseClustererSelection(Enum):
    PARTITIONAL_SPACE = auto()
    KMEANS_SPACE = auto()
    ALL_ALGOS_SPACE = auto()


class ConsensusFunction:
    def __init__(self, cc_function, name):
        self.cc_function = cc_function
        self.name = name

    def execute_consensus(self, Y, k_out):
        return self.cc_function(Y, k_out)

    def get_name(self):
        return self.name


IVC = ConsensusFunction(iterative_voting_consensus, "IVC")
QMI = ConsensusFunction(qmi, "QMI")
ABV = ConsensusFunction(ada_bvote, "ABV")
ACV = ConsensusFunction(ada_cvote, "ACV")
MCLA = ConsensusFunction(mcla, "MCLA")
MM = ConsensusFunction(mm, "MM")

CC_FUNCTIONS = [
    MCLA,
    # IVC,
    QMI,
    ABV,
    ACV,
    MM
]

CC_function_mapping: dict[str, type(ConsensusFunction)] = {cc.get_name(): cc for cc in CC_FUNCTIONS}


@memory(percentage=1.5)
def execute_consensus_clustering(cc_name, Y, k_out: int):
    try:
        if isinstance(Y, np.matrix):
            Y = np.asarray(Y)
        return CC_function_mapping[cc_name].execute_consensus(Y, k_out), ""

    except Exception as e:
        print(e)
        print("Exception occurred - return default result")
        return [0] * Y.shape[0], e


def build_consensus_cs(cc_functions: list[ConsensusFunction] = CC_FUNCTIONS,
                       k_range: [int, int] = (2, 100),
                       max_ensemble_size: Union[int, None] = 50,
                       step_size: Union[int, None] = 5,
                       default_ensemble_size: Union[int, None] = 5,
                       exclude=None  # TODO: Leave out some of ccs
                       ) -> ConfigurationSpace:
    """
    Args:
        cc_functions: List of consensus functions. Must have the name of the consensus function as key, and the value has to be a function that takes as input the esnemble and the 'k' value.
        k_range: Range of possible k values, e.g., (2,100). Set both values equally to have a single k value, e.g., k_range=(2,2).
        max_ensemble_size: If the max. ensemble size is set, then this indicates that the ensemble size will be optimized as well!
        step_size:Specifies the grid-size for the ensemble size. Of course, we could also use all from (2,max_ensemble_size). This is equivalent to step_size=1.
        default_ensemble_size: This parameter should only be set when max_ensemble_size is not defined.
         This means that a default value is used and the ensemble size will not be optimized.        exclude:

    Returns:
        ConfigurationSpace object

    """
    cs = ConfigurationSpace()
    # if len(cc_functions.items()) == 1:

    cc_function_hp = CategoricalHyperparameter("cc_function",
                                               default_value=cc_functions[0].get_name(),
                                               choices=[cc_function.get_name()
                                                        for cc_function in cc_functions])
    cs.add_hyperparameter(cc_function_hp)

    if k_range[0] == k_range[1]:
        k_hyperparameter = Constant("k", value=k_range[0])
    else:
        k_hyperparameter = UniformIntegerHyperparameter("k", lower=k_range[0], upper=k_range[1],
                                                        default_value=k_range[0])
    cs.add_hyperparameter(k_hyperparameter)

    # cs.add_condition(InCondition(k_hyperparameter, cc_function_hp, list(cc_functions.keys())))

    if max_ensemble_size:
        ensemble_size_hp = UniformIntegerHyperparameter("m", lower=step_size, upper=max_ensemble_size,
                                                        default_value=step_size,
                                                        q=step_size)
    else:
        ensemble_size_hp = Constant("m", value=default_ensemble_size)
    cs.add_hyperparameter(ensemble_size_hp)

    return cs


def build_consensus_cs_default_ens_size(cc_functions=CC_FUNCTIONS, k_range=(2, 100),
                                        default_ensemble_size=5,
                                        ):
    return build_consensus_cs(cc_functions=cc_functions, k_range=k_range,
                              default_ensemble_size=default_ensemble_size,
                              max_ensemble_size=None, step_size=None)


def build_consensus_cs_for_cf(single_cc_function=CC_FUNCTIONS[0],
                              k_range=(2, 100),
                              default_ensemble_size=None,
                              max_ensemble_size=30,
                              step_size=5):
    # single_cc_function_mapping = {single_cc_function: CC_function_mapping[single_cc_function]}
    if default_ensemble_size:
        return build_consensus_cs_default_ens_size(cc_functions=[single_cc_function],
                                                   k_range=k_range,
                                                   default_ensemble_size=default_ensemble_size)
    else:
        return build_consensus_cs(cc_functions=[single_cc_function],
                                  k_range=k_range,
                                  max_ensemble_size=max_ensemble_size,
                                  step_size=step_size,
                                  default_ensemble_size=None)


if __name__ == "__main__":
    cs = build_consensus_cs(CC_FUNCTIONS,
                            k_range=(2, 2))
    print(cs)

    cs = build_consensus_cs_default_ens_size(default_ensemble_size=5)
    print(cs)
