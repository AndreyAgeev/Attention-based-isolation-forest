from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

import numpy as np
from omegaconf import DictConfig

from atif.logger import FileLogger


class Mode(Enum):
    CLASSIC = 0,
    ATTENTION = 1


class AbstractModel(ABC):
    """Abstract class for model."""
    def __init__(self, *args, **kwargs):
        self._logger: Optional[FileLogger] = None

    def setup_logger(self, logger: FileLogger):
        self._logger = logger

    def setup(self, cfg: DictConfig):
        pass

    @abstractmethod
    def fit(self, data: np.ndarray, labels: np.ndarray, improved: bool):
        pass

    @abstractmethod
    def predict(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    @abstractmethod
    def optimize_weights(self, data, labels):
        pass

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def score_samples(self, data):
        pass

    @abstractmethod
    def get_type_model(self):
        pass

    @abstractmethod
    def set_param_isolation_forest(self, offset):
        pass

    @abstractmethod
    def set_param_attention(self, eps, sigma, softmax_tau):
        pass

    @abstractmethod
    def create_tree(self, seed, n_estimators, max_samples):
        pass
