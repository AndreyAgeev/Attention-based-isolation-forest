import pandas as pd
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8


class DatasetIonosphere(AbstractDataset):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def load(self):
        """https://www.kaggle.com/code/zymzym/classification-of-the-ionosphere-dataset-by-knn"""
        df = pd.read_csv(self._path)
        df.drop(columns=['column_b'], inplace=True)
        df.rename(columns={'column_ai': 'label'}, inplace=True)
        df['label'] = df.label.astype('category')
        encoding = {'g': 0, 'b': 1}
        df.label.replace(encoding, inplace=True)
        fraud_data = df[df['label'] == 1]
        norm_data = df[df['label'] == 0]
        data, labels = df.drop('label', axis=1), df['label']

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "ionosphere"

