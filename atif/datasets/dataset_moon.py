import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset
import numpy as np


class DatasetMoon(AbstractDataset):
    def __init__(self, n_samples_normal: int, n_samples_outliners: int, rng: int):
        super().__init__()
        self._n_samples_normal = n_samples_normal
        self._n_samples_outliners = n_samples_outliners
        self._rng = rng

    def load(self):
        self._create()

    def _create(self):
        rng = np.random.RandomState(42)
        # X, y = make_moons(n_samples=self._n_samples_normal, noise=0.05, random_state=0)
        X = 14.0 * (np.random.RandomState(42).rand(self._n_samples_normal, 2) - 0.5),
        # X = 5.0 * (X - np.array([0.5, 0.25]))

        # X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(self._n_samples_outliners, 2))], axis=0)

        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=0))
        # df[df['label'] == 1] = 0
        outlier_data_plus = rng.uniform(low=-6, high=6, size=(self._n_samples_outliners, 2))
        df_outlier_plus = pd.DataFrame({'x': outlier_data_plus[:, 0], "y": outlier_data_plus[:, 1], 'label': 1})
        df_combined = df.append(df_outlier_plus)
        data, labels = df_combined.drop('label', axis=1), df_combined['label']
        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                            test_size=0.33, random_state=42) #maybe random state

        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        plt.scatter(data[:, 0], data[:, 1], c=y, s=20, edgecolor="k")
        plt.savefig(save_path)
        plt.clf()

    def get_name(self) -> str:
        return "moon=" + str(self._n_samples_normal) + "_anomal=" + str(self._n_samples_outliners)
