from EnsMetaLearning.EffEns import EffEns
from Experiments.SyntheticData import DataGeneration

if __name__ == "__main__":
    datasets = DataGeneration.generate_datasets(n_values=[1000,
                                                          # 5000,
                                                          10000, 50000])

    d_names = datasets.keys()
    X_list = [v[0] for v in datasets.values()]
    y_list = [v[1] for v in datasets.values()]

    effEns = EffEns(k_range=(2, 100), random_state=1234)
    effEns.run_learning_phase(dataset_names=d_names,
                              X_list=X_list, y_list=y_list,
                              store_result=False)
