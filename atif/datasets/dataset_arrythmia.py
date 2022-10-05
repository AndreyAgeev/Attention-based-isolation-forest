import pandas as pd
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DatasetArrythmia(AbstractDataset):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def load(self):
        """https://www.kaggle.com/code/mtavares51/binary-classification-on-arrhythmia-dataset"""
        target = "diagnosis"
        df = pd.read_csv(self._path, sep=';')
        df.dropna(axis=0, inplace=True)
        df.drop(df.columns[20:-2], axis=1, inplace=True)
        df.drop(['T', 'P', 'J', 'LG'], axis=1, inplace=True)
        j = []
        for i in df.diagnosis:
            if i in [3, 4, 5, 7, 8, 9, 14, 15]:
                j.append(1)
            else:
                j.append(0)
        df.diagnosis = j

        def label_encoding(old_column):
            le = LabelEncoder()
            le.fit(old_column)
            new_column = le.transform(old_column)
            return new_column

        # encoding string parameters
        for i in df.columns:
            if type(df[i][0]) == str:
                df[i] = label_encoding(df[i])

        # extracting x and y
        labels = df[target].values
        fraud_data = df[df[target] == 1]
        norm_data = df[df[target] == 0]

        data = df.drop([target], axis=1).values
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "ionosphere"
