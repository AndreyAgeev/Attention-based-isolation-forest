from omegaconf import DictConfig
from hydra.utils import instantiate
from collections import defaultdict

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc
import numpy as np

from atif.core import AbstractModel, AbstractDataset, Mode
from atif.logger import FileLogger


class Solution:
    def __init__(self, cfg: DictConfig):
        self._model: AbstractModel = instantiate(cfg.model.type_model, _recursive_=False)
        self._dataset: AbstractDataset = instantiate(cfg.dataset, _recursive_=False)
        self._logger = FileLogger()
        self._setup(cfg)
        self._table_res = None
        self._report_path = cfg.report_path
        # self._offset_variants = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        # self._eps_variants = [0.0, 0.25, 0.5, 0.75, 1.0]
        # self._sigma_variants = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        # self._softmax_tau = [0.1, 10, 20, 30, 40]
        self._offset_variants = [0.3, 0.4, 0.5, 0.6]
        self._eps_variants = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._sigma_variants = [0.5, 0.6, 0.7]
        self._softmax_tau = [0.1, 10, 20, 30, 40]
        self._seed_variants = [1234 + 7 * i for i in range(30)]

    def _setup(self, cfg: DictConfig):
        self._logger.setup(cfg.logger)
        self._dataset.load()
        self._model.setup(cfg.model)
        self._n_estimators = cfg.model.n_estimators

    def run(self):
        all_F1 = defaultdict(list)
        max_f1, best_offset, best_seed = 0.0, None, None
        max_f1, best_eps, best_sigma, best_seed = 0.0, None, None, None

        if self._model.get_type_model() == Mode.CLASSIC:
            for seed in self._seed_variants:
                self._model.create_tree(seed, self._n_estimators, len(self._dataset.X_train))
                self._create_table_if()
                self._model.fit(self._dataset.X_train,
                                self._dataset.y_train)
                for offset in self._offset_variants:
                    self._model.set_param_isolation_forest(offset)
                    y_prediction = self._model.predict(self._dataset.X_test, self._dataset.y_test)
                    F1 = f1_score(self._dataset.y_test, y_prediction)
                    if F1 > max_f1:
                        max_f1 = F1
                        best_offset = offset
                        best_seed = seed
                    all_F1[offset].append(F1)
                    print("offset = ", offset)
            for offset in self._offset_variants:
                self._table_res.loc[offset] = np.mean(all_F1[offset])
            self._table_res.to_csv(self._report_path +
                                   f"if_estimators={self._n_estimators}"
                                   f"_offset={best_offset}"
                                   f"_best_seed={best_seed}_max_f1={max_f1}.csv",
                                   index=True, header=True, sep=' ')
        else:
            for softmax_tau in self._softmax_tau:
                for seed in self._seed_variants:
                    self._model.create_tree(seed, self._n_estimators, len(self._dataset.X_train))
                    self._model.fit(self._dataset.X_train,
                                    self._dataset.y_train)
                    self._create_table_abif()
                    for eps in self._eps_variants:
                        for sigma in self._sigma_variants:
                            self._model.set_param_attention(eps, sigma, softmax_tau)
                            self._model.optimize_weights(self._dataset.X_train,
                                                         self._dataset.y_train)
                            y_prediction = self._model.predict(self._dataset.X_test, self._dataset.y_test)
                            F1 = f1_score(self._dataset.y_test, y_prediction)
                            if F1 > max_f1:
                                max_f1 = F1
                                best_sigma = sigma
                                best_eps = eps
                                best_seed = seed
                            all_F1[(eps, sigma)].append(F1)
                for eps in self._eps_variants:
                    for sigma in self._sigma_variants:
                        self._table_res.loc[sigma, eps] = np.mean(all_F1[(eps, sigma)])
                self._table_res.to_csv(self._report_path +
                                       f"abif_estimators={self._n_estimators}"
                                       f"softmax_tau={softmax_tau}"
                                       f"_eps={best_eps}"
                                       f"_sigma={best_sigma}"
                                       f"_best_seed={best_seed}_max_f1={max_f1}.csv",
                                       index=True, header=True, sep=' ')

    def _create_table_if(self):
        import pandas as pd
        A = np.random.randint(1, size=len(self._offset_variants))
        df = pd.DataFrame(A, index=self._offset_variants, columns=["F1"])
        self._table_res = df
        # df.to_csv(self._report_path + "if_df.csv", index=True, header=True, sep=' ')

    def _create_table_abif(self):
        import numpy as np
        import pandas as pd

        A = np.random.randint(0, 1, size=len(self._sigma_variants) * len(self._eps_variants))\
            .reshape(len(self._sigma_variants), len(self._eps_variants))

        df = pd.DataFrame(A, index=self._sigma_variants, columns=self._eps_variants)
        self._table_res = df

    def _get_F1(self, y_prediction):
        return f1_score(self._dataset.y_test, y_prediction)

    def close(self):
        self._logger.end_logger("end model inference")


def optimization(cfg: DictConfig):
    """Main function.

    Args:
        cfg (DictConfig): config structure by .yaml.
    """
    print("wtf")
    solution_runner = Solution(cfg)
    solution_runner.run()
    solution_runner.close()
