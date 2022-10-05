import random
from typing import Any, Optional, Tuple
from enum import Enum

import cvxpy as cp
import numpy as np
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from scipy.special import softmax as _softmax

from atif.core import AbstractModel, Mode


def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0


class AttentionBasedIsolationForestSklearn(AbstractModel):
    def __init__(self,
                 mode: int = 0):
        super().__init__()
        self._clf = None
        self._original_if_offset = None
        self._n_estimators = None
        self._eps = None
        self._softmax_tau = None
        self._attention_sigma_threshold = None
        self._sigma = None
        self._w = None
        if mode == 0:
            self._mode = Mode.CLASSIC
        else:
            self._mode = Mode.ATTENTION

    def create_tree(self, seed, n_estimators, max_samples):
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)
        self._n_estimators = n_estimators
        self._clf = IsolationForest(n_estimators=n_estimators,
                                    max_samples="auto",
                                    random_state=rng) #seed для некоторых передавал

    def setup(self, cfg: DictConfig):
        if self._mode == Mode.CLASSIC:
            self._original_if_offset = -cfg.isolation_forest_param.offset
        else:
            self._eps = cfg.attention_param.eps
            self._softmax_tau = cfg.attention_param.softmax_tau
            self._attention_sigma_threshold = cfg.attention_param.attention_sigma_threshold

    def set_param_isolation_forest(self, offset):
        self._original_if_offset = -offset

    def set_param_attention(self, eps, sigma, softmax_tau):
        self._eps = eps
        self._attention_sigma_threshold = sigma
        self._softmax_tau = softmax_tau

    def fit(self, data: np.ndarray, labels: np.ndarray, improved: bool = False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        self._clf.fit(data)
        # self._original_if_offset = self._clf.offset_
        # if self._mode == Mode.ATTENTION:
        #     self._w = np.ones(self._n_estimators) / self._n_estimators
        #     self.optimize_weights(data, labels)
        return self

    def optimize_weights(self, data, labels):
        self._w = np.ones(self._n_estimators) / self._n_estimators
        labels = np.array([-1 if labels[i] == 0 else labels[i] for i in range(len(labels))])
        self._prepare_leaf_data(data, labels)
        dynamic_weights, dynamic_x, dynamic_y = self._get_dynamic_weights_y(data)
        static_weights = cp.Variable((1, self._n_estimators))
        sigma = -c(len(data)) * np.log2(self._attention_sigma_threshold)
        self._sigma = sigma
        y = labels
        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]
            mixed_weights = (1.0 - self._eps) * dynamic_weights + self._eps * static_weights
            print("Shapes 1:", mixed_weights.shape, dynamic_y.shape)
        else:
            dynamic_y = dynamic_y.reshape((dynamic_y.shape[0], -1))
            n_trees = dynamic_y.shape[1]
            n_outs = dynamic_y.shape[0]
            y = y.reshape((-1))
            dynamic_weights = np.tile(dynamic_weights[:, np.newaxis, :], (1, n_outs, 1)).reshape((-1, n_trees))
            mixed_weights = (1.0 - self._eps) * dynamic_weights + self._eps * static_weights
            print("Shapes 2:", mixed_weights.shape, dynamic_y.shape)
        # y = np.array([-1 if y[i] == 0 else y[i] for i in range(len(y))])
        z = cp.pos(y * (cp.sum(cp.multiply(mixed_weights, dynamic_y), axis=1)
                        - self._sigma))
        min_obj = cp.sum(z)
        reg = cp.norm(static_weights, 1)
        # lambd = cp.Parameter(nonneg=True)

        problem = cp.Problem(cp.Minimize(min_obj + 0.0 * reg),
                             [
                                 static_weights >= 0,
                                 cp.sum(static_weights, axis=1) == 1,
                                 # z >= 0,
                                 # z >= y * (cp.sum(cp.multiply(mixed_weights, dynamic_y), axis=1)
                                 #           - sigma)
                             ]
                             )
        try:
            loss_value = problem.solve(qcp=True)
        except Exception as ex:
            print(f"Solver error: {ex}")

        if static_weights.value is None:
            print(f"Can't solve problem with OSQP. Trying another solver...")
            loss_value = problem.solve(solver=cp.SCS, qcp=True, verbose=True)

        if static_weights.value is None:
            print(f"Weights optimization error (eps={self._eps}). Using default values.")
        else:
            self._w = static_weights.value.copy().reshape((-1,))
            print("Weights = ", self._w)
        return self

    def anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return self._clf.score_samples(data)

    def predict(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        if self._mode == Mode.CLASSIC:
            # threshold, _ = self._find_TPR_threshold(labels, self._clf.score_samples(data), 1.0)
            self._clf.offset_ = self._original_if_offset
            predictions = self._clf.predict(data)
            predictions[predictions == 1] = 0
            predictions[predictions == -1] = 1
        else:
            dynamic_weights, dynamic_x, dynamic_y = self._get_dynamic_weights_y(data)
            mixed_weights = (1.0 - self._eps) * dynamic_weights + self._eps * self._w
            dynamic_y = dynamic_y[..., 0]
            predictions = np.sign((np.sum(mixed_weights * dynamic_y, axis=1) - self._sigma))
            predictions[predictions == 1] = 0
            predictions[predictions == -1] = 1
        return predictions

    def _compute_depth(self, X, tree_index):
        """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.

        subsample_features : bool
            Whether features should be subsampled.
        """
        n_samples = X.shape[0]

        depths = np.zeros(n_samples, order="f")
        tree = self._clf.estimators_[tree_index]
        features = self._clf.estimators_features_[tree_index]
        X_subset = X[:, features]

        leaves_index = tree.apply(X_subset)
        node_indicator = tree.decision_path(X_subset)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        depths += (
            np.ravel(node_indicator.sum(axis=1))
            + _average_path_length(n_samples_leaf)
            - 1.0
        )
        return depths

    def _prepare_leaf_data_fast(self, xs, y, leaf_ids, estimators):
        """Utility function for preparing forest tree leaf data.
        For each leaf tree finds all corresponding training samples,
        and calculates averages for input vectors and target values.
        Args:
            xs: Input vectors.
            y: Target values.
            leaf_ids: Input samples to leaves ids correspondence
                      (see `sklearn.ensemble.RandomForestClassifier.apply`).
            estimators: List of estimators.
        Returns:
            A pair of leaf data for input vectors and target values.
        """
        max_leaf_id = max(map(lambda e: e.tree_.node_count, estimators))
        y_len = 1 if y.ndim == 1 else y.shape[1]
        result_x = np.full((len(estimators), max_leaf_id + 1, xs.shape[1]), np.nan, dtype=np.float32)
        result_y = np.full((len(estimators), max_leaf_id + 1, y_len), np.nan, dtype=np.float32)
        for tree_id in range(len(estimators)):
            for leaf_id in range(estimators[tree_id].tree_.node_count + 1):
                mask = (leaf_ids[tree_id, :] == leaf_id)
                masked_xs = xs[mask]
                masked_y = y[mask]
                if True in mask:
                    masked_y = self._compute_depth(masked_xs, tree_id) # проверить, что точки одинаковые, раз они в один лист идут
                if mask.any():
                    result_x[tree_id, leaf_id] = masked_xs.mean(axis=0)
                    result_y[tree_id, leaf_id] = masked_y.mean(axis=0) #ВОТ ТУТ НЕ НАДО MEAN брать, надо проверить что тут правильно длина берется в конкретном дереве а не средлняя длина в лесе

        return result_x, result_y

    def _prepare_leaf_data(self, data, label):
        data = data.astype('float32')
        training_leaf_ids = []
        for t in range(self._n_estimators):
            training_leaf_ids.append(self._clf.estimators_[t].tree_.apply(data)) # индекс листа для каждого объекта
            # fig = plt.figure(figsize=(25, 20))
            # tree.plot_tree(self._clf.estimators_[t])
            # fig.savefig("decistion_tree" + str(t) + ".png")
        training_leaf_ids = np.array(training_leaf_ids)
        self.leaf_data_x, self.leaf_data_y = self._prepare_leaf_data_fast(
            data,
            label,
            np.array(training_leaf_ids),
            self._clf.estimators_,
        )

    def _get_dynamic_weights_y(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        leaf_ids = []
        data = data.astype('float32')
        for t in range(self._n_estimators):
            leaf_ids.append(self._clf.estimators_[t].tree_.apply(data))
        leaf_ids = np.array(leaf_ids)
        all_dynamic_weights = []
        all_y = []
        all_x = []
        for cur_index, cur_x in enumerate(data):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            tree_dynamic_x = []
            for cur_tree_id, cur_leaf_id in enumerate(leaf_ids[:, cur_index]):
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]#
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]#
                tree_dynamic_weight = -0.5 * (np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0)
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
                tree_dynamic_x.append(leaf_mean_x)
            tree_dynamic_weights = _softmax(np.array(tree_dynamic_weights) * self._softmax_tau)
            tree_dynamic_y = np.array(tree_dynamic_y)
            tree_dynamic_x = np.array(tree_dynamic_x)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_x.append(tree_dynamic_x)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_x, all_y

    def get_name(self) -> str:
        if self._mode == Mode.ATTENTION:
            return "attention_n_estimators=" + str(self._n_estimators) + "_eps=" + str(self._eps) \
                   + "_attention_sigma_threashold=" + str(self._attention_sigma_threshold) \
                   + "_softmax_tau=" + str(self._softmax_tau)
        else:
            return "classic_n_estimators=" + str(self._n_estimators) + "_offset=" + str(self._original_if_offset)

    def score_samples(self, data):
        if self._mode == Mode.CLASSIC:
            return -self._clf.score_samples(data)
        else:
            dynamic_weights, dynamic_x, dynamic_y = self._get_dynamic_weights_y(data)
            mixed_weights = (1.0 - self._eps) * dynamic_weights + self._eps * self._w
            dynamic_y = dynamic_y[..., 0]
            return np.sum(mixed_weights * dynamic_y, axis=1) - self._sigma

    def get_type_model(self):
        return self._mode
