import pandas as pd
from sklearn.model_selection import train_test_split

from atif.core import AbstractDataset

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing


class DatasetHttp(AbstractDataset):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def load(self):
        """https://www.openml.org/search?type=data&sort=runs&id=40897&status=active"""
        """https://github.com/dple/Datasets"""
        df = pd.read_csv(self._path)
        #
        # df.loc[df["Target"] == "Anomaly", "Target"] = 1
        # df.loc[df["Target"] == "Normal", "Target"] = 0
        # label_encoder = preprocessing.LabelEncoder()
        # df['Target'] = label_encoder.fit_transform(df['Target'])
        # # df = df.dropna(axis=0)
        # df['Target'] = df['Target'].map({0: 1, 1: 0})

        # df.loc[df["Target"] == 0, "Target"] = 1
        # df.loc[df["Target"] == 1, "Target"] = 0
        # df['Target'] = df['Target'].astype(int)
        # df['Target'] = df['Target'].map({'Anomaly': '1', 'Normal': '0'})
        fraud = len(df[df['attack'] == 1])
        valid = len(df[df['attack'] == 0])
        fraud_data = df[df['attack'] == 1].iloc[0:50]
        norm_data = df[df['attack'] == 0].iloc[0:500]
        df = norm_data.append(fraud_data)
        # df['Target'] = df['Target'].str.replace(',', '')
        # df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        # df['Target'] = df['Target'].astype('str')
        # encoding = {"Anomaly": 1, "Normal": 0}
        # df['Target'].replace(encoding, inplace=True)
        data, labels = df.drop('attack', axis=1), df['attack']
        # col = data.columns
        # std = StandardScaler()
        # x = std.fit_transform(data)
        # data = pd.DataFrame(data=x, columns=col)
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
        return "http"

