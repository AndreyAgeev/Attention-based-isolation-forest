from abc import ABC, abstractmethod
from typing import List


class AbstractMetric(ABC):
    """Abstract class for metrics."""
    @abstractmethod
    def process(self, target: List, score: List):
        """Process function."""
        pass

    @abstractmethod
    def calculate(self) -> List:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass
