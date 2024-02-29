import argparse
import time
from pathlib import Path

from ConfigSpace.configuration_space import Configuration
from scipy.stats import spearmanr
import ast

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ClusteringCS import ClusteringCS
from MetaLearning import MetaFeatureExtractor, DataGeneration
from MetaLearning.MetaFeatureExtractor import load_kdtree, query_kdtree, extract_meta_features
from Metrics.MetricHandler import MetricCollection
from Optimizer.OptimizerSMAC import SMACOptimizer
import pandas as pd
import numpy as np
import joblib
import seaborn as sns

from Utils.Helper import mf_set_to_string

sns.set(style="darkgrid")
aml4c_path = Path("/volume/aml4c")

# define random seed
np.random.seed(1234)

##############################################################################
############################### Parameter Specification ######################
parser = argparse.ArgumentParser()
parser.add_argument("--initialization", help="Option for warmstarting.",
                    default="warmstart", choices=["warmstart", "coldstart"])
parser.add_argument("--metric", help="Option for learning an optimization metric.",
                    default="learning", choices=["learning"] + MetricCollection.internal_metrics)
parser.add_argument("--dataset_type", help="Option for running on specific dataset types. "
                                           "Per default, all synthetic dataset types are used.",
                    nargs='+', default=DataGeneration.DATASET_TYPES,
                    choices=DataGeneration.DATASET_TYPES + ['all'])
parser.add_argument("--phase", help="Option for running either offline or online phase. "
                                    "Per default, Online phase is executed.",
                    default='online',
                    choices=['offline', 'online'])
parser.add_argument("--n_warmstarts", help="Specify number of warmstarting configurations. Should only be used if"
                                           " initialization is warmstart. Default are 25 warmstarts", default=25)
parser.add_argument("--n_loops", help="Specifies the number of configurations/optimizer loops that should be executed."
                                      " Default are 100 loops", default=100)
parser.add_argument("-limit_cs", help="Defines wether to limit the config space according to the best found"
                                      " configurations from the offline phase.", default=False, action='store_true')
parser.add_argument("--time_limit", help="Defines the runtime of the optimization procedure."
                                         " Per default the time limit is 2 hours (120 * 60 s)", default=120 * 60)
args = parser.parse_args()

initialization = args.initialization
initialization = "warmstart"
metric_learning = args.metric
use_classification_model = True

dataset_types = args.dataset_type
if dataset_types == ['all']:
    dataset_types = DataGeneration.DATASET_TYPES

n_warmstarts = args.n_warmstarts
n_loops = args.n_loops
limit_cs = args.limit_cs
time_limit = args.time_limit
execution_phase = args.phase

execution_phase = "online"
use_large_datasets = True
##############################################################################

limit_cs = True
if limit_cs:
    # put this experiment in a new directory
    aml4c_path = aml4c_path / "limit_cs"

if not Path(aml4c_path).exists():
    Path(aml4c_path).mkdir()


def run_coldstart():
    online_opt_result_all_datasets = pd.DataFrame()
    # todo: dont reuse results
    if Path(aml4c_path / f'coldstart_online_result.csv').is_file() and False:
        online_opt_result_all_datasets = pd.read_csv(aml4c_path / f'coldstart_online_result.csv', index_col=None)
        if "Unnamed: 0" in online_opt_result_all_datasets.columns:
            online_opt_result_all_datasets.drop('Unnamed: 0', axis='columns')
        if "Unnamed: 0.1" in online_opt_result_all_datasets.columns:
            online_opt_result_all_datasets.drop('Unnamed: 0.1', axis='columns')

    # for the moment we use some of the metrics --> DBI, SIL, CH, DBCV
    coldstart_metrics = MetricCollection.internal_metrics

    for coldstart_metric in coldstart_metrics:
        shape_sets = DataGeneration.generate_datasets(dataset_types=dataset_types)

        datasets_to_use = [dataset[0] for key, dataset in shape_sets.items()]
        dataset_names_to_use = list(shape_sets.keys())
        true_labels_to_use = [dataset[1] for key, dataset in shape_sets.items()]

        print("Using datasets: ")

        for dataset, dataset_name, dataset_labels, in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
            cs = ClusteringCS.build_all_algos_space()
            metric_abbrev = coldstart_metric.get_abbrev()
            if 'dataset' in online_opt_result_all_datasets.columns:
                if len(online_opt_result_all_datasets[
                           (online_opt_result_all_datasets['dataset'] == dataset_name)
                           & (online_opt_result_all_datasets['metric'] == metric_abbrev)]) >= 1:
                    print(f"Dataset {dataset_name} already processed with metric {metric_abbrev}")
                    continue
            else:
                print(f"Dataset {dataset_name} not! processed with metric {metric_abbrev}")

            dataset = StandardScaler().fit_transform(dataset)

            opt_instance = SMACOptimizer(dataset=dataset, true_labels=dataset_labels,
                                         metric=coldstart_metric,
                                         n_loops=n_loops, cs=cs, wallclock_limit=time_limit)

            opt_instance.optimize()
            online_opt_result_df = opt_instance.get_runhistory_df()

            online_opt_result_df['iteration'] = [i + 1 for i in range(len(online_opt_result_df))]

            # We have the metric name as column. However, we want a coulmn with metric and then the name of that metric
            online_opt_result_df['metric'] = metric_abbrev
            online_opt_result_df['metric score'] = online_opt_result_df[metric_abbrev].cummin()
            online_opt_result_df['wallclock time'] = online_opt_result_df['runtime'].cumsum()

            # set max_iteration --> need this to check for timeouts
            max_iteration = online_opt_result_df['iteration'].max()
            max_wallclock_time = online_opt_result_df['wallclock time'].max()
            online_opt_result_df['max wallclock'] = max_wallclock_time
            online_opt_result_df['max iteration'] = max_iteration

            # we only want to look at, where the incumbent changes! so where we get better metric values. Hence, the result
            # will not contain all performed iterations!
            # todo: Do this at analysis/plotting if necessary!
            # todo: this will use ALL Ari scores if not here --> we only want best ones
            online_opt_result_df = online_opt_result_df[
                online_opt_result_df[metric_abbrev] == online_opt_result_df['metric score']]

            online_opt_result_df['ARI'] = [MetricCollection.ADJUSTED_RAND.score_metric(data=None, labels=labels,
                                                                                       true_labels=dataset_labels) for
                                           labels
                                           in online_opt_result_df['labels']]
            iterations = online_opt_result_df["iteration"].values
            if len(iterations > 0):
                print(iterations)
                last_iteration = 0
                for i in range(1, 101):
                    if i in iterations:
                        last_iteration = i
                    else:
                        it_filtered = online_opt_result_df[
                            online_opt_result_df["iteration"] == last_iteration]
                        it_filtered["iteration"] = i
                        print(it_filtered)
                        online_opt_result_df = pd.concat([online_opt_result_df, it_filtered])

            online_opt_result_df = online_opt_result_df.drop([metric_abbrev, 'labels', 'budget'],
                                                             axis=1)

            print(f"ARI scores are: {online_opt_result_df['ARI']}")
            online_opt_result_df['dataset'] = dataset_name

            print(online_opt_result_df.head())
            print(online_opt_result_df.columns)

            online_opt_result_all_datasets = pd.concat([online_opt_result_all_datasets, online_opt_result_df])
            online_opt_result_all_datasets.to_csv(aml4c_path / f'coldstart_online_result.csv', index=False)


def load_meta_knowledge_run_warmstart():
    ex_to_data_to_ranking = {'mf_set': [],
                             'dataset': [],
                             'metric': [],
                             'ranking': [],
                             'correlation': [],
                             'optimal metric': [],
                             'metric_ranking': [],
                             'most similar dataset': [],
                             'algorithms': []}

    metric_for_data_for_ex = pd.DataFrame(columns=["dataset", "metric", "mf_set", "optimal"])
    for mf_set in MetaFeatureExtractor.meta_feature_sets:

        online_opt_result_all_datasets = pd.DataFrame()

        shape_sets = DataGeneration.generate_datasets(dataset_types=dataset_types)

        datasets_to_use = [dataset[0] for key, dataset in shape_sets.items()]
        dataset_names_to_use = list(shape_sets.keys())
        true_labels_to_use = [dataset[1] for key, dataset in shape_sets.items()]

        print("Using datasets: ")

        for dataset, dataset_name, dataset_labels, in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
            # 1. load optimizer knowledge --> Independent of extraction method
            opt_results_df = pd.read_csv(aml4c_path / 'opt_meta_knowledge.csv', index_col=0)

            print(f"Using dataset to query: {dataset_name}")
            # 2. Extract meta_features
            t0 = time.time()
            names, meta_features = extract_meta_features(dataset, mf_set)
            mf_time = time.time() - t0

            n_similar_datasets = 1

            # 3. Load kdtree
            tree = load_kdtree(path=aml4c_path, mf_set=mf_set)

            # 4. find nearest neighbors
            dists, inds = query_kdtree(meta_features, tree, k=len(d_names))

            # dists, inds = query_kdtree(meta_features, tree, k=72)
            inds = inds[0]
            dists = dists[0]

            # Prepare Warmstart
            most_similar_dataset_names = [d_names[ind] for ind in inds]
            opt_results_df = opt_results_df[opt_results_df['dataset'].isin(most_similar_dataset_names)]

            # assign algorithm column
            opt_results_df["algorithm"] = opt_results_df.apply(lambda x: ast.literal_eval(x["config"])["algorithm"],
                                                               axis="columns")

            # print(dists)
            # assign distance column
            dataset_name_to_distance = {d_name: dists[ind] for ind, d_name in enumerate(most_similar_dataset_names)}

            opt_results_df['distance'] = [dataset_name_to_distance[dataset_name] for dataset_name
                                          in opt_results_df['dataset']]

            # filter such that the same dataset is not used. Note that for mf extraction, the same dataset does
            # not necessarily have distance=0
            opt_results_for_dataset = opt_results_df[opt_results_df['dataset'] != dataset_name]
            opt_results_for_dataset = opt_results_for_dataset.sort_values(['distance'], ascending=[True])

            # Duplikate in den configs entfernen --> Wir behalten die config vom ähnlichsten Datensatz
            opt_results_for_dataset = opt_results_for_dataset.drop_duplicates(subset='config', keep='first')

            # only look at top n_warmstarts configs of each dataset for warmstarting
            top_warmstart_results = pd.concat(
                [group[1].sort_values('ARI', ascending=True)[0:n_warmstarts] for group in
                 list(opt_results_for_dataset.groupby('distance'))[0:n_similar_datasets]])

            optimal_metrics = pd.read_csv(f"{aml4c_path}/optimal_metric.csv")
            optimal_metric = optimal_metrics[optimal_metrics["dataset"] == dataset_name]["metric"].values[0]
            if use_classification_model:
                # todo: other classifiers can be used as well
                rf: RandomForestClassifier = joblib.load(
                    f"{aml4c_path}/models/RandomForestClassifier/{mf_set_to_string(mf_set)}/{dataset_name}")
                best_metric = rf.predict(meta_features.reshape(1,-1))[0]
                print(f"best metric for {dataset_name} is: {best_metric}")
            else:

                # take top 100 results for metric ranking
                metric_ranking_results = pd.concat([group[1].sort_values('ARI', ascending=True)[0:100] for group in
                                                    list(opt_results_for_dataset.groupby('distance'))[
                                                    0:n_similar_datasets]])

                print(top_warmstart_results.head())
                print(f"most similar datasets for {dataset_name}: {top_warmstart_results['dataset'].unique()}")
                most_similar_dataset = top_warmstart_results['dataset'].values[0]
                metric_ranking_results = metric_ranking_results.fillna(0)

                n_metric_ranking_configs = 20

                ######################################################################################################
                ###################################### CAlculate Spearman Correlation for each metric#################
                # get for each internal metric the best result --> We do not want to execute the optimizer loop yet
                correlation_by_metric = {}

                for internal_metric in MetricCollection.internal_metrics:
                    top_metric_result = metric_ranking_results[
                        [internal_metric.get_abbrev(), 'ARI', 'distance',
                         'runtime', 'algorithm', 'config']]

                    top_metric_result = top_metric_result.sort_values(internal_metric.get_abbrev(), ascending=True)[
                                        0:n_metric_ranking_configs]

                    spearman_correlation, p_value = spearmanr(top_metric_result['ARI'],
                                                              top_metric_result[internal_metric.get_abbrev()],
                                                              nan_policy='omit'
                                                              )

                    # spearman_correlation = np.mean(top_metric_result.sort_values(internal_metric.get_abbrev(), ascending=True)['ARI'].values[0:20])

                    if np.isnan(spearman_correlation) & (len(top_metric_result['ARI'].unique()) == 1) \
                            & (top_metric_result['ARI'].values[0] == -1.0):
                        spearman_correlation = 1

                    correlation_by_metric[internal_metric.get_abbrev()] = spearman_correlation
                    # sort ascending because they should be minimized!
                    top_metric_result = top_metric_result.sort_values(internal_metric.get_abbrev(), ascending=True)

                    print("--------------------------------------")
                    print(f"------{internal_metric.get_abbrev()}----------------")
                    print(top_metric_result[[internal_metric.get_abbrev(), "config", "ARI"]])
                    print("-----------------spearman---------------------")
                    print(f"Spearman correlation is: {spearman_correlation}")

                # obtain the metric that has the best correlation
                best_metric = max(correlation_by_metric, key=correlation_by_metric.get)
                # best_metric = min(correlation_by_metric, key=correlation_by_metric.get)

                print(correlation_by_metric)

                print(f"best metric is: {best_metric}")
                metric_ranking = {key: rank for rank, key in
                                  enumerate(sorted(correlation_by_metric, key=correlation_by_metric.get, reverse=True),
                                            1)}
                print(f"our metric ranking: {metric_ranking}")
                print(correlation_by_metric)
                print("-------------------------------------------------")
                print("-------------------------------------------------")
                ######################################################################################################
                ######################################################################################################

                ######################################################################################################
                ###############################CAlculate Optimal Spearman Correlation for each metric#################
                # we have to do the same for the results with the dataset that is used! So we get the best "optimal" metric
                # (configs?)
                unseen_data_opt_results_df = opt_results_df[opt_results_df['dataset'] == dataset_name]
                optimal_correlation_by_metric = {}
                print(opt_results_df)

                for internal_metric in MetricCollection.internal_metrics:
                    optimal_ranking_for_dataset = unseen_data_opt_results_df[
                        [internal_metric.get_abbrev(), 'ARI', 'distance', 'algorithm']]
                    # sort for ARI rank
                    optimal_ranking_for_dataset = optimal_ranking_for_dataset.sort_values(
                        [internal_metric.get_abbrev()], ascending=True)[
                                                  0:n_metric_ranking_configs]
                    print(optimal_ranking_for_dataset)

                    # checking without custom ranking
                    spearman_correlation, p_value = spearmanr(optimal_ranking_for_dataset['ARI'],
                                                              optimal_ranking_for_dataset[internal_metric.get_abbrev()],
                                                              nan_policy='omit'
                                                              )
                    if np.isnan(spearman_correlation) & (len(optimal_ranking_for_dataset['ARI'].unique()) == 1) \
                            & (optimal_ranking_for_dataset['ARI'].values[0] == -1.0):
                        spearman_correlation = 1

                    optimal_correlation_by_metric[internal_metric.get_abbrev()] = spearman_correlation

                    print("--------------------------------------")
                    print(f"------{internal_metric.get_abbrev()}----------------")
                    print("-----------------spearman---------------------")
                    print(f"Spearman correlation is: {spearman_correlation}")

                # obtain the metric that has the best correlation
                optimal_metric = max(optimal_correlation_by_metric, key=optimal_correlation_by_metric.get)
                print(f"optimal metric is: {optimal_metric}")
                print(optimal_correlation_by_metric)
                metric_for_data_for_ex = metric_for_data_for_ex.append({"dataset": dataset_name,
                                                                        "mf_set": mf_set_to_string(mf_set),
                                                                        "metric": best_metric,
                                                                        "optimal": optimal_metric},
                                                                       ignore_index=True)
                print(metric_for_data_for_ex)

                # should actually be optimal_correlation_by_metric[key] == 1 and not correlation_by_metric
                # However, if we have correlation = 1 for a metric, this should be ok for the dataset!
                optimal_metric_ranking = {key: 1 if correlation_by_metric[key] == 1 else rank for rank, key in
                                          enumerate(
                                              sorted(optimal_correlation_by_metric,
                                                     key=optimal_correlation_by_metric.get,
                                                     reverse=True), 1)}
                print(f"optimal metric ranking: {optimal_metric_ranking}")
                ######################################################################################################
                ######################################################################################################

            # ex_to_data_to_ranking = {'ex_name': [], 'dataset': [], 'metric': [], 'ranking': [], 'correlation': []}
            ex_to_data_to_ranking['mf_set'].append(mf_set_to_string(mf_set))
            ex_to_data_to_ranking['dataset'].append(dataset_name)
            ex_to_data_to_ranking['metric'].append(best_metric)
            ex_to_data_to_ranking['optimal metric'].append(optimal_metric)
            # ex_to_data_to_ranking['ranking'].append(optimal_metric_ranking[best_metric])
            # ex_to_data_to_ranking['correlation'].append(correlation_by_metric[best_metric])
            #ex_to_data_to_ranking['most similar dataset'].append(most_similar_dataset)
            #ex_to_data_to_ranking['metric_ranking'].append(correlation_by_metric)
            #ex_to_data_to_ranking['algorithms'].append(metric_ranking_results['algorithm'].unique())

            # todo: put continue here if you want to skip the warmstart but only want to look at metric_ranking
            #continue
            ####################################################################################################
            ############################ Run Warmstart Optimization Procedure ##################################
            # Now we have the best metric chosen and the warmstart configs are also there as well!
            # These are the 'config' from top_opt_results, which are the results for the 'other' datasets.
            cs = ClusteringCS.build_all_algos_space()

            if initialization == "warmstart":
                print(opt_results_for_dataset.sort_values(['distance', 'ARI'], ascending=[True, True])[
                          ['ARI', 'distance']][0:n_warmstarts])
                # warmstart_configs = opt_results_for_dataset.drop_duplicates(subset=['ARI', 'algorithm']).sort_values(['distance', 'ARI'], ascending=[True, True])['config'][0:n_warmstarts]
                warmstart_configs = top_warmstart_results[
                                        'config'][0:n_warmstarts]
                print(f"number of warmstart configs: {len(warmstart_configs)}")
                print(warmstart_configs)
                # the configs are saved as strings, so we need ast.literal_eval to convert them to dictionaries
                warmstart_configs = [ast.literal_eval(config_string) for config_string in warmstart_configs]
                if limit_cs:
                    # retrieve algorithms from the warmstart configs as list without duplicates
                    algorithms = list({d['algorithm'] for d in warmstart_configs})
                    print(algorithms)
                    algorithms = list(top_warmstart_results['algorithm'].unique())
                    print(algorithms)

                    # Now we update the config space to only contain the "best" algorithms
                    cs = ClusteringCS.build_config_space(clustering_algorithms=algorithms, dim_reductions=None)
                    print(cs)

                # now create the configuration objects which are passed to the optimizer
                warmstart_configs = [Configuration(cs, config_dict) for config_dict in warmstart_configs]

            else:
                warmstart_configs = []
                print(warmstart_configs)

            # get the metric object from "best_metric" which will be passed to the optimizer
            if metric_learning == "learning":
                best_internal_metric = MetricCollection.get_metric_by_abbrev(best_metric)
            else:
                best_internal_metric = MetricCollection.CALINSKI_HARABASZ

            opt_instance = SMACOptimizer(dataset=dataset, true_labels=dataset_labels,
                                         metric=best_internal_metric,
                                         # n_loops=25,
                                         n_loops=n_loops,
                                         cs=cs, wallclock_limit=time_limit
                                         )

            opt_instance.optimize(initial_configs=warmstart_configs)
            online_opt_result_df = opt_instance.get_runhistory_df()

            # We have the metric name as column. However, we want a coulmn with metric and then the name of that metric
            online_opt_result_df['iteration'] = [i + 1 for i in range(len(online_opt_result_df))]

            online_opt_result_df['metric'] = best_internal_metric.get_abbrev()
            online_opt_result_df['metric score'] = online_opt_result_df[best_internal_metric.get_abbrev()].cummin()
            online_opt_result_df['wallclock time'] = online_opt_result_df['runtime'].cumsum()

            # set max_iteration --> need this to check for timeouts
            max_iteration = online_opt_result_df['iteration'].max()
            max_wallclock_time = online_opt_result_df['wallclock time'].max()
            online_opt_result_df['max wallclock'] = max_wallclock_time
            online_opt_result_df['max iteration'] = max_iteration

            # we only want to look at, where the incumbent changes! so where we get better metric values. Hence, the result
            # will not contain all performed iterations!
            online_opt_result_df = online_opt_result_df[
                online_opt_result_df[best_internal_metric.get_abbrev()] == online_opt_result_df['metric score']]
            online_opt_result_df['ARI'] = [MetricCollection.ADJUSTED_RAND.score_metric(data=None, labels=labels,
                                                                                       true_labels=dataset_labels) for
                                           labels
                                           in online_opt_result_df['labels']]

            ### Add the iterations that are not present at the moment due to the fact that we only take best ARI scores####
            ### i.e., we only want to have the same result for iterations  2-10 as for iteration 1 if the incumbent does not change ###
            iterations = online_opt_result_df["iteration"].values
            if len(iterations > 0):
                print(iterations)
                last_iteration = 0
                for i in range(1, 101):
                    if i in iterations:
                        last_iteration = i
                    else:
                        it_filtered = online_opt_result_df[online_opt_result_df["iteration"] == last_iteration]
                        it_filtered["iteration"] = i
                        print(it_filtered)
                        online_opt_result_df = pd.concat([online_opt_result_df, it_filtered])

            online_opt_result_df = online_opt_result_df.drop([best_internal_metric.get_abbrev(), 'labels', 'budget'],
                                                             axis=1)
            print(online_opt_result_df.columns)
            print(online_opt_result_df[online_opt_result_df["iteration"] <= 25][
                      ["ARI", "iteration", "config", "metric score", ]])

            print(online_opt_result_df['ARI'])
            online_opt_result_df['dataset'] = dataset_name
            online_opt_result_df['mf time'] = mf_time

            print(online_opt_result_df.head())
            print(online_opt_result_df.columns)
            #exit()
            online_opt_result_all_datasets = pd.concat([online_opt_result_all_datasets, online_opt_result_df])
            online_opt_result_all_datasets.to_csv(aml4c_path / f'{mf_set_to_string(mf_set)}_online_result.csv')

    # metric_for_data_for_ex.to_csv(aml4c_path / "metrics.csv")
    pd.DataFrame(ex_to_data_to_ranking).to_csv(aml4c_path / 'metric_ranking.csv')


different_shape_sets = DataGeneration.generate_datasets()

d_names = list(different_shape_sets.keys())
datasets = [X for X, y in different_shape_sets.values()]
true_labels = [y for X, y in different_shape_sets.values()]
print(len(d_names))
print(len(datasets))

# Extract indices in sorted order for each
indices_per_extraction = {}
distances_per_extraction = {}

if metric_learning and initialization == 'warmstart':
    load_meta_knowledge_run_warmstart()
else:
    run_coldstart()
