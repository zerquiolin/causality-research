from abc import ABC, abstractmethod
from typing import List


class BaseMetric(ABC):
    @abstractmethod
    def evaluate(self) -> tuple[str, list]:
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )


class BehaviorMetric(BaseMetric):
    name: str

    @abstractmethod
    def evaluate(self, history) -> float:
        """Evaluate this behavior metric on the provided history."""
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )


class DeliverableMetric(BaseMetric):
    name: str

    @abstractmethod
    def evaluate(self, scm, data) -> float:
        """Evaluate this deliverable metric on the provided history."""
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )


# === Utility ===


class WeightedScores:
    def __init__(self, values: List[float], weights: List[float]):
        assert len(values) == len(weights), "Values and weights must match length"
        self.values = values
        self.weights = weights

    def compute(self) -> float:
        return sum(v * w for v, w in zip(self.values, self.weights))
