import ast
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from automlclustering.ClusterValidityIndices import CVIHandler
from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from ConsensusCS import ConsensusCS
from ConsensusCS.ConsensusCS import CC_function_mapping, CC_FUNCTIONS, build_consensus_cs, ConsensusFunction
from EffEnsMKR import MKR
from Utils.Utils import get_type_from_dataset
from EnsOptimizer import EnsembleSelection
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from EnsOptimizer.cas import Cas
from EnsMetaLearning import MetaFeatureExtractor

cc_functions = CC_function_mapping.keys()


def train_cf_classifier(X_metafeatures_train, y_cf_train, random_state=1234):
    clf = RandomForestClassifier(n_estimators=1000,
                                 random_state=random_state)
    clf.fit(X_metafeatures_train, y_cf_train)
    return clf


def _append_dummies(df_train, y_train):
    df_train["tmp"] = y_train

    df_train = df_train.join(pd.get_dummies(df_train["tmp"])).drop("tmp", axis=1)
    # Corner Case for CC Functions
    if "MCLA" in y_train:
        # Maybe one of the CFs is not predicted -> We get one column less for one-hot encoding
        for cc_function in cc_functions:
            if cc_function not in df_train.columns:
                df_train[cc_function] = 0
    return df_train


def _prepare_k_ensemble_training_data(df_mfs_train, best_eval_ensemble, append_cfs=True):
    k_range = (2, 101)
    k_in_ensemble = {k: [] for k in range(k_range[0], k_range[1] + 1)}

    # Generate n X len(k_range) matrix -> '1' if k value is in the ensemble otherwise '0'
    for j in range(best_eval_ensemble.shape[0]):
        row = best_eval_ensemble.iloc[j, :]
        k_values = ast.literal_eval(row["Ensemble"])

        for k in range(k_range[0], k_range[1] + 1):
            if k in k_values:
                k_in_ensemble[k].append(1)
            else:
                k_in_ensemble[k].append(0)

    multi_label_targets = pd.DataFrame(k_in_ensemble)

    if append_cfs:
        # Append Consensus Function as separate input column to the model
        df_mfs_cf_train = _append_dummies(df_mfs_train, best_eval_ensemble["cc_function"].values)
    else:
        df_mfs_cf_train = df_mfs_train
    return df_mfs_cf_train, multi_label_targets


def predict_consensus_function(mf_new, cf_model):
    cf_pred = cf_model.predict(mf_new.reshape(1, -1))[0]
    print(cf_pred)
    # Assign cc functions columns
    mf_new_df = pd.DataFrame(mf_new)
    for cf in cc_functions:
        mf_new_df[cf] = 0

    # Add a one for the predicted CF
    mf_new_df[cf_pred] = 1
    return cf_pred, mf_new_df


class EffEns:

    def __init__(self, path_to_mkr=MKR.path_to_mkr, k_range=(2, 100),
                 random_state=0,
                 mf_set=MetaFeatureExtractor.meta_feature_sets[5]
                 ):

        self.mf_set = mf_set
        self.logger = logging.Logger(self.__class__.__name__)

        assert len(k_range) == 2
        assert k_range[0] < k_range[1]

        self.k_range = k_range

        self.path_to_mkr = Path(path_to_mkr)
        self.random_state = random_state

        self.EGM = None
        self.CFM = None
        self.mf_scaler = None
        # Train both models in either way, because there is no much overhead
        # EGM, CFM, mf_scaler = self._train_EGM_CFM_model()
        # self.EGM = EGM
        # self.CFM = CFM
        # self.mf_scaler = mf_scaler

    def apply_ensemble_clustering(self, X, cvi, ensemble="EGM", consensus_functions="CFM",
                                  ensemble_k_heuristic=[2, 100], n_loops=70):

        if not self.EGM and not self.CFM:
            # Train Models if we have not already set them
            # Todo: Modelle laden? Ggfs. sogar immer das beste aus der Lernphase nehmen?
            EGM, CFM, mf_scaler = self._train_EGM_CFM_model()
            self.EGM = EGM
            self.CFM = CFM
            self.mf_scaler = mf_scaler
        cvi = self._validate_cvi(cvi)

        mf_new = MetaFeatureExtractor.extract_meta_features(X, self.mf_set)[1].reshape(1, -1)
        # Apply the learned scaler
        mf_new = self.mf_scaler.transform(mf_new)
        additional_result_info = {"cvi": cvi.get_abbrev()}

        if consensus_functions == "CFM":
            cf_pred, mf_new_df = predict_consensus_function(mf_new, self.CFM)
            additional_result_info["cf"] = cf_pred
        else:
            cf_pred = consensus_functions

        additional_result_info["cf"] = cf_pred

        gen_time = time.time()
        if ensemble == "EGM":
            pred_k_values = self.EGM.predict(mf_new_df.to_numpy())[0]
            k_values_ens = [k + self.k_range[0] for k, value in enumerate(pred_k_values) if value == 1]

            for k in ensemble_k_heuristic:
                if k not in k_values_ens:
                    k_values_ens.append(k)

            # generate ensemble
            ens = np.zeros((len(k_values_ens), X.shape[0]))
            for i, k in enumerate(k_values_ens):
                km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=self.random_state)
                y_k = km.fit_predict(X)
                ens[i, :] = y_k
            ens = ens.transpose()
            self.logger.log(msg=f"k-values in the ensemble: {k_values_ens}", level=logging.INFO)

        else:
            # User can also provide custom ensemble
            ens = ensemble
        gen_time = time.time() - gen_time
        additional_result_info["gen_time"] = gen_time

        consensus_optimizer = self._run_ensemble_optimization(X, cf_pred, ens, cvi, n_loops=n_loops)
        return consensus_optimizer, additional_result_info

    def _run_ensemble_optimization(self, X, cf_pred, ens, cvi, n_loops=70):
        cc_functions_to_use = self._validate_cf_pred(cf_pred)

        # 6.3) Build consensus cs
        consensus_cs = build_consensus_cs(k_range=self.k_range,
                                          cc_functions=cc_functions_to_use,
                                          default_ensemble_size=ens.shape[1],
                                          max_ensemble_size=None)

        # 6.4) Optimization
        ens_optimizer = EnsembleOptimizer(dataset=X,
                                          generation_cvi=cvi,
                                          cs_generation=consensus_cs,
                                          cs_consensus=consensus_cs,
                                          seed=self.random_state)
        ens_optimizer.ensemble = ens
        self.ensemble = ens
        ens_optimizer.optimize_consensus(n_loops=n_loops, k_range=self.k_range)
        return ens_optimizer.consensus_optimizer

    def get_ensemble(self):
        if self.ensemble is not None:
            return self.ensemble
        else:
            self.logger.log(level=logging.WARN, msg="No ensemble, first run optimization to set the ensemble")
            return None

    def _train_EGM_CFM_model(self):
        self.logger.log(msg="Start loading meta-features and evaluated ensembles ...", level=logging.INFO)
        mfs_df = pd.read_csv(Path(self.path_to_mkr) / MKR.path_to_mkr_meta_features)
        mfs_df = mfs_df.sort_values("dataset")
        dataset_names = mfs_df["dataset"].unique()
        mfs_df = mfs_df[mfs_df["dataset"].isin(dataset_names)]

        eval_ens = pd.read_csv(Path(self.path_to_mkr) / MKR.path_to_eval_ens)
        eval_ens["cc_function"] = eval_ens["config"].apply(lambda x: ast.literal_eval(x)["cc_function"])
        eval_ens["type"] = eval_ens["dataset"].apply(get_type_from_dataset)
        eval_ens = eval_ens[eval_ens["dataset"].isin(dataset_names)]
        eval_ens = eval_ens.sort_values("dataset")

        self.logger.log(msg="Finished", level=logging.INFO)

        self.logger.log(msg="Start scaling meta-feature values...", level=logging.INFO)
        # scale meta-feature values
        X_mfs_train = mfs_df.drop("dataset", axis=1).to_numpy()
        mf_scaler = StandardScaler()
        mf_scaler.fit(X_mfs_train)
        X_mfs_train = mf_scaler.transform(X_mfs_train)
        self.logger.log(msg="Finished", level=logging.INFO)

        self.logger.log(msg="Start finding best CF for each dataset and best performing ensemble", level=logging.INFO)
        best_eval_ensemble = pd.DataFrame()
        # iterate over all datasets in MKR
        for dataset in eval_ens["dataset"].unique():
            # results for this dataset
            dataset_eval_ens = eval_ens[eval_ens["dataset"] == dataset]

            # Results with highest ARI values
            best_dataset_ari = dataset_eval_ens[dataset_eval_ens["ARI"] == dataset_eval_ens["ARI"].min()]

            # Use some heuristics if we have multiple best CF functions
            if ("moons" in dataset or "circles" in dataset) and "MCLA" in best_dataset_ari["cc_function"]:
                best_eval_ensemble = pd.concat([best_eval_ensemble,
                                                best_dataset_ari[best_dataset_ari["cc_function"] == "MCLA"]])

            elif "ACV" in best_dataset_ari["cc_function"]:
                best_eval_ensemble = pd.concat([best_eval_ensemble,
                                                best_dataset_ari[best_dataset_ari["cc_function"] == "ACV"]])
            else:
                best_eval_ensemble = pd.concat([best_eval_ensemble,
                                                best_dataset_ari[
                                                    best_dataset_ari["runtime"] == best_dataset_ari["runtime"].min()]])

        # Best results for each dataset in MKR
        best_eval_ensemble = best_eval_ensemble.sort_values("dataset")

        self.logger.log(msg="Finished", level=logging.INFO)
        # We should have only one result for each dataset!
        assert len(best_eval_ensemble) == len(best_eval_ensemble["dataset"].unique())

        ##################################################
        ### Training Consensus Function Model (CFM) ######
        # Best Consensus Function for each dataset
        y_cf_train = [cf for cf in best_eval_ensemble["cc_function"].values]
        self.logger.log(msg="start training CFM...", level=logging.INFO)
        # Train Classifier Model based on meta-features
        cf_model = train_cf_classifier(X_mfs_train, y_cf_train, random_state=self.random_state)
        self.logger.log(msg="Finished training CFM", level=logging.INFO)

        ##################################################
        #### Training Ensemble Generation Model (EGM) ####
        df_mfs_train = pd.DataFrame(X_mfs_train)

        df_mfs_cf_train, multi_label_targets = _prepare_k_ensemble_training_data(df_mfs_train,
                                                                                 best_eval_ensemble)

        ensemble_k_model = RandomForestClassifier(n_estimators=1000, random_state=self.random_state,
                                                  # min_samples_leaf=10 --> Leads to m= 0 in any case
                                                  )
        self.logger.log(msg="start training EGM...", level=logging.INFO)
        ensemble_k_model.fit(df_mfs_cf_train.to_numpy(), multi_label_targets)
        self.logger.log(msg="Finished training EGM", level=logging.INFO)

        return ensemble_k_model, cf_model, mf_scaler

    @staticmethod
    def _validate_cvi(cvi):
        if isinstance(cvi, str):
            return CVIHandler.CVICollection.get_cvi_by_abbrev(cvi)
        elif isinstance(cvi, CVIHandler.CVI):
            return cvi
        else:
            raise ValueError(f"CVI type unknown, should either be str or CVI class. Got: {cvi} ({type(cvi)})")

    @staticmethod
    def _validate_cf_pred(cf_pred):
        if not cf_pred:
            # Per default, return all CC functions
            return [CC_function_mapping[cf.get_name()] for cf in CC_FUNCTIONS]
        if isinstance(cf_pred, list):
            if np.alltrue([isinstance(cf, ConsensusFunction) for cf in cf_pred]) and np.alltrue(
                    [cf in CC_FUNCTIONS for cf in cf_pred]):
                return cf_pred
            else:
                raise ValueError(f"Unkown list of consensus functions {cc_functions}")
        elif isinstance(cf_pred, str):
            if cf_pred in CC_function_mapping.keys():
                return [CC_function_mapping[cf_pred]]

            else:
                raise ValueError(f"Unknown string representation of Consenus Function {cf_pred}, should be one of "
                                 f"{CC_function_mapping.keys()}")
        elif isinstance(cf_pred, ConsensusFunction):
            return [cf_pred]
        else:
            raise ValueError(f"Unkown Consenus Function {cf_pred}, should be one of {CC_FUNCTIONS}")

    def get_name(self):
        return self.__class__.__name__

    def run_learning_phase(self, dataset_names, X_list, y_list, cvi=CVICollection.ADJUSTED_MUTUAL, store_result=True):
        # L1: Extract Meta-Features
        mf_results = self._extract_meta_features(dataset_names, X_list, store_results=store_result)

        # L2 + L3: Generate Base Clusterings and evaluate ensembles
        evaluated_ensembles = self._gen_and_eval_ensembles(dataset_names, X_list, y_list, store_result, cvi=cvi)

        # L4: Select best CF and Ensemble + L5: Train EGM and CFM models
        EGM, CFM, mf_scaler = self._train_EGM_CFM_model()
        self.EGM = EGM
        self.CFM = CFM
        # We also need the scaler model of the meta-feature values
        self.mf_scaler = mf_scaler

    def _extract_meta_features(self, dataset_names, X_list, store_results=True):
        mf_results = pd.DataFrame()

        for d_name, X in zip(dataset_names, X_list):
            mfs = MetaFeatureExtractor.extract_meta_features(X, mf_set=self.mf_set)
            mf_df = pd.DataFrame(mfs)
            mf_df["dataset"] = d_name
            mf_results = pd.concat([mf_results, mf_df])

            if store_results:
                mf_results.to_csv(self.path_to_mkr / "meta_features.csv")

        return mf_results

    def _gen_and_eval_ensembles(self, dataset_names, X_list, y_list, store_result, cvi=CVICollection.ADJUSTED_MUTUAL):
        result_df = pd.DataFrame()

        for d_name, X, y in zip(dataset_names, X_list, y_list):
            print(f"Running on dataset: {d_name}")

            # L4: Generate Exhaustive Ensemble
            sorted_gen_ensemble = self._run_exhaustive_ensemble_generation(cvi=cvi)

            # L5: Evaluate Ensembles
            dataset_result = self._evaluate_ensembles(d_name, sorted_gen_ensemble, y, store_result)

            # Store result
            result_df = pd.concat([result_df, dataset_result])

        return result_df

    def _run_exhaustive_ensemble_generation(self, cvi):
        ensemble = np.zeros((self.k_range[1] - self.k_range[0], X.shape[0]))
        cvi_values = np.zeros(self.k_range[1] - self.k_range[0])

        print(f"Running Ensemble Generation")
        for k in range(self.k_range[0], self.k_range[1]):
            print(f"Running k-Means with k={k}")
            start = time.time()
            km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=self.random_state)
            y_pred = km.fit_predict(X)
            ensemble[k - self.k_range[0], :] = y_pred
            print(f"Executed kmeans - took {time.time() - start}s")
            print(f"Start scoring {cvi.get_abbrev()}")
            start = time.time()
            cvi_values[k - self.k_range[0]] = cvi.score_cvi(data=X, labels=y_pred, true_labels=y)
            print(f"Finished scoring {cvi.get_abbrev()}, took {time.time() - start}s")

        print(f"Finished Ensemble Generation")
        gen_ensemble = ensemble.transpose()
        sorted_gen_ensemble = gen_ensemble[:, cvi_values.argsort()]
        return sorted_gen_ensemble

    def _evaluate_ensembles(self, d_name, sorted_gen_ensemble, y, store_result):
        result_df = pd.DataFrame()
        for es_selection in [EnsembleSelection.SelectionStrategies.cluster_and_select,
                             EnsembleSelection.SelectionStrategies.quality
                             ]:
            actual_k = np.unique(y)
            if es_selection == EnsembleSelection.SelectionStrategies.cluster_and_select:
                print(f"Using CAS")
                # precompute NMI matrix, do it only once and not in every step, because it is very time-consuming!
                cas = Cas(sorted_gen_ensemble)

            ensemble_sizes = list(range(5, 55, 5))

            for m in ensemble_sizes:
                consensus_cs = build_consensus_cs(k_range=self.k_range, default_ensemble_size=m,
                                                  max_ensemble_size=None)
                print(consensus_cs)

                if es_selection == EnsembleSelection.SelectionStrategies.quality:

                    selected_ens = sorted_gen_ensemble[:, 0:m]
                    selected_ens = selected_ens.astype(int)

                elif es_selection == EnsembleSelection.SelectionStrategies.cluster_and_select:
                    cas_start = time.time()
                    selected_ens = cas.cluster_and_select(base_labels=sorted_gen_ensemble, m=m)
                    print(f"Took {time.time() - cas_start}s")

                # Determine k values that we use in the ensemble
                ensemble_k_values = []
                for i in range(selected_ens.shape[1]):
                    k_in_ens = len(np.unique(selected_ens[:, i]))
                    ensemble_k_values.append(k_in_ens)

                assert len(ensemble_k_values) == selected_ens.shape[1]

                for cf in ConsensusCS.CC_FUNCTIONS:
                    cf_result = {}
                    cf_start = time.time()
                    y_pred = cf.execute_consensus(selected_ens, k_out=actual_k)
                    cf_runtime = time.time() - cf_start

                    cf_result["runtime"] = cf_runtime
                    cf_result["config"] = str({"cc_function": cf.get_name(), "k": actual_k, "m": m})
                    cf_result["ARI"] = -1 * adjusted_rand_score(y, y_pred)
                    cf_result["NMI"] = -1 * normalized_mutual_info_score(y, y_pred)
                    cf_result["AMI"] = -1 * adjusted_mutual_info_score(y, y_pred)
                    cf_result["es_selection"] = es_selection.value
                    cf_result["dataset"] = d_name
                    cf_result["Ensemble"] = str(ensemble_k_values)
                    result_df = pd.concat([result_df, pd.DataFrame({k: [v] for k, v in cf_result.items()})],
                                          ignore_index=True)
                    print(result_df)
                    if store_result:
                        print("Storing result")
                        result_df.to_csv(self.path_to_mkr / "evaluated_ensemble.csv", index=False)
        return result_df


if __name__ == '__main__':
    X, y = make_blobs(n_samples=10000, centers=30, n_features=100)

    effEns = EffEns(k_range=(2, 100))
    consensus_opt, add_info = effEns.apply_ensemble_clustering(X, cvi=CVIHandler.CVICollection.CALINSKI_HARABASZ,
                                                               n_loops=5)
    print(consensus_opt.get_run_history())
    print(consensus_opt.get_incumbent_stats())
    y_pred = consensus_opt.get_incumbent_stats()["labels"]
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

    print(f"ARI score: {adjusted_rand_score(y, y_pred)}")
