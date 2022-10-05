import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from atif.core import AbstractDataset
from sklearn import preprocessing

from pylab import rcParams
rcParams['figure.figsize'] = 14, 8


class DatasetCredit(AbstractDataset):
    def __init__(self, path: str, target_col: str = None):
        super().__init__()
        self._path = path
        self._target_col = target_col

    def load(self):
        """https://www.kaggle.com/code/shivamsekra/credit-card-fraud-detection-eda-isolation-forest"""
        data = pd.read_csv(self._path)
        label_encoder = preprocessing.LabelEncoder()
        data[self._target_col] = label_encoder.fit_transform(data[self._target_col])
        fraud_data = data[data['Class'] == 1].iloc[0:400]
        norm_data = data[data['Class'] == 0].iloc[0:1500]
        data = norm_data.append(fraud_data)

        # stroing the normal and fraud cases

        # self.fraud = len(df[df[self._target_col] == 1])
        # self.valid = len(df[df[self._target_col] == 0])
        count_class = pd.value_counts(data['Class'])
        count_class.plot(kind='bar', rot=0)
        plt.title("Class Distribution")
        plt.xticks(range(2), ["Normal", "Fraud"])
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        # plt.savefig("/Users/andreyageev/PycharmProjects/ATIF/image/credit_dataset.jpg")

        fraud = data[data['Class'] == 1]
        normal = data[data['Class'] == 0]
        print(fraud.shape, normal.shape)
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.suptitle('Amount per transaction by class')
        bins = 50
        ax1.hist(fraud.Amount, bins=bins)
        ax1.set_title('Fraud')
        ax2.hist(normal.Amount, bins=bins)
        ax2.set_title('Normal')
        plt.xlabel('Amount ($)')
        plt.ylabel('Number of Transactions')
        plt.xlim((0, 20000))
        plt.yscale('log')

        data, labels = data.drop(self._target_col, axis=1), data[self._target_col]
        from sklearn.preprocessing import RobustScaler

        rob_scaler = RobustScaler()
        data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
        data.drop(['Time', 'Amount'], axis=1, inplace=True)
        scaled_amount = data['scaled_amount']
        scaled_time = data['scaled_time']
        data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        data.insert(0, 'scaled_amount', scaled_amount)
        data.insert(1, 'scaled_time', scaled_time)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values

    def plot_dataset(self, data, y, save_path):
        pass

    def get_name(self) -> str:
        return "credit=" + str(self.valid) + "_anomal=" + str(self.fraud)

