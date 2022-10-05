import pandas as pd
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


class DatasetPimaDiabetes(AbstractDataset):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def load(self):
        """https://www.kaggle.com/code/hafizramadan/data-science-project-iii"""
        df = pd.read_csv(self._path)
        fraud = len(df[df['Outcome'] == 1])
        valid = len(df[df['Outcome'] == 0])
        data, labels = df.drop('Outcome', axis=1), df['Outcome']
        col = data.columns
        std = StandardScaler()
        x = std.fit_transform(data)
        data = pd.DataFrame(data=x, columns=col)
        # over = RandomOverSampler(random_state=42)
        # data, labels = over.fit_resample(data, labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                            test_size=0.33, random_state=42)
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "diabetes"

