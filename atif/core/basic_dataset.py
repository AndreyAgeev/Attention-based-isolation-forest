from abc import ABC, abstractmethod

import numpy as np


class AbstractDataset(ABC):
    """Abstract class for dataset."""
    def __init__(self):
        self.X_train: np.ndarray = []
        self.X_test: np.ndarray = []
        self.y_train: np.ndarray = []
        self.y_test: np.ndarray = []

    @abstractmethod
    def load(self):
        """Process function."""
        pass

    @abstractmethod
    def plot_dataset(self, data, y, save_path):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
