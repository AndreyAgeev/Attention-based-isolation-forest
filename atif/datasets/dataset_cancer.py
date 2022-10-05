import pandas as pd
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset


class DatasetCancer(AbstractDataset):
    def __init__(self, path: str, target_col: str = None):
        super().__init__()
        self._path = path
        self._target_col = target_col

    def load(self):
        df = pd.read_csv(self._path)
        fraud_data = df[df[self._target_col] == 1].iloc[0:400]
        norm_data = df[df[self._target_col] == 0].iloc[0:1500]
        df = norm_data.append(fraud_data)
        data, labels = df.drop(self._target_col, axis=1), df[self._target_col]
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "cancer"
