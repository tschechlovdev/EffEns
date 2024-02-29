from typing import Callable, Iterable, List

from .adjusted_ba_kmeans import adjusted_ba_kmeans as qmi
from .cvs import ada_bvote, ada_cvote
from .ivc import iterative_voting_consensus
from .mcla_py_wrapper import mcla
from .mixture_model import mm
from .simple_functions import choose_best_automl

all_consensus_functions: List[Callable] = [qmi, iterative_voting_consensus, ada_cvote, ada_bvote, mcla, mm]
all_consensus_functions_aml: Iterable[Callable] = all_consensus_functions + [choose_best_automl]
all_consensus_functions_no_vote: Iterable[Callable] = [qmi, iterative_voting_consensus, mcla, mm]
all_all_consensus_functions: Iterable[Callable] = [qmi, iterative_voting_consensus, ada_cvote, mcla, mm]
all_no_vote: Iterable[Callable] = [qmi, iterative_voting_consensus, mcla, mm]
all_no_vote_no_ivc: Iterable[Callable] = [qmi, mcla, mm]

consensus_functions_display_names = {
    qmi.__name__: "QMI",
    iterative_voting_consensus.__name__: "IVC",
    ada_cvote.__name__: "A-CV",
    ada_bvote.__name__: "A-BV",
    mcla.__name__: "MCLA",
    mm.__name__: "MM",
    choose_best_automl.__name__: "A4C-1"
}

consensus_functions_display_names_to_function_names = {
    "QMI": qmi.__name__,
    "IVC": iterative_voting_consensus.__name__,
    "ADA-CVOTE": ada_cvote.__name__,
    "ADA-BVOTE": ada_bvote.__name__,
    "MCLA": mcla.__name__,
    "MM": mm.__name__,
    "BEST-AML": choose_best_automl.__name__
}
