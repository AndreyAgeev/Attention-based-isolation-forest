import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from atif.core import AbstractDataset

import matplotlib.pyplot as plt


class DatasetGenerated1D(AbstractDataset):
    def __init__(self, n_samples_normal: int, n_samples_outliners: int, rng: int):
        super().__init__()
        self._n_samples_normal = n_samples_normal
        self._n_samples_outliners = n_samples_outliners
        self._rng = rng

    def load(self):
        self._create()

    def _create(self):
        rng = np.random.RandomState(42)

        # Generate train data
        # X = 0.3 * rng.randn(100, 2)
        # X_train = np.r_[X + 2, X - 2]
        # # Generate some regular novel observations
        # X = 0.3 * rng.randn(20, 2)
        # X_test = np.r_[X + 2, X - 2]
        # # Generate some abnormal novel observations
        # X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))


        # Generate train data
        X = 0.6 * rng.randn(self._n_samples_normal, 2)
        standard_data = np.r_[X + 2, X - 2]
        # standard_data = np.r_[X + 2, X + 2]

        # Generate some abnormal novel observations
        outlier_data_plus = rng.uniform(low=-0.9, high=0.9, size=(self._n_samples_outliners, 2))

        # standard_data = np.random.randn(self._n_samples_normal, 2)
        # outlier_data_plus = np.random.uniform(low=5, high=10, size=(self._n_samples_outliners, 2))
        df = pd.DataFrame({'x1': standard_data[:, 0], "x2": standard_data[:, 1], 'Distribution': 0})
        df_outlier_plus = pd.DataFrame({'x1': outlier_data_plus[:, 0], "x2": outlier_data_plus[:, 1], 'Distribution': 1})
        df_combined = df.append(df_outlier_plus)
        data, labels = df_combined.drop('Distribution', axis=1), df_combined['Distribution']

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42) #maybe random state

        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        plt.scatter(data[:, 0], data[:, 1], c=y, s=20, edgecolor="k")
        plt.savefig(save_path)
        plt.clf()

    def get_name(self) -> str:
        return "generated_dataset_normal=" + str(self._n_samples_normal) + "_anomal=" + str(self._n_samples_outliners)
