import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset


class DatasetSklearn(AbstractDataset):
    def __init__(self, n_samples_normal: int, n_samples_outliners: int, rng: int):
        super().__init__()
        self._n_samples_normal = n_samples_normal
        self._n_samples_outliners = n_samples_outliners
        self._rng = rng

    def load(self):
        self._create()

    def _create(self):
        X, y = make_circles(n_samples=self._n_samples_normal * 2, factor=0.5, noise=0.1, random_state=42)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df_ok = pd.DataFrame({
            'x1': df[df["label"] == 1]["x"],
            "x2": df[df["label"] == 1]["y"],
            'Distribution': 0})
        df_outlier_plus = pd.DataFrame(
            {'x1': df[df["label"] == 0]["x"].iloc[0:self._n_samples_outliners],
             "x2": df[df["label"] == 0]["y"].iloc[0:self._n_samples_outliners],
             'Distribution': 1})
        df_combined = df_ok.append(df_outlier_plus)
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
        return "circle=" + str(self._n_samples_normal) + "_anomal=" + str(self._n_samples_outliners)
