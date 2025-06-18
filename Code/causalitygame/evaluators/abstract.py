# Abstract
from abc import ABC, abstractmethod

# Types
from typing import Any


class BaseMetric(ABC):
    """
    Abstract base class providing a common interface for all metrics.
    """

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> float:
        """
        Evaluate the metric.

        Returns:
            float: The metric result.
        """


class BehaviorMetric(BaseMetric):
    """
    Base class for behavior-based metrics.
    """

    @abstractmethod
    def evaluate(self, history: Any) -> float:
        """
        Evaluate this behavior metric on the provided history.

        Args:
            history: Historical data of actions or events.

        Returns:
            float: The behavior metric score.
        """


class DeliverableMetric(BaseMetric):
    """
    Base class for deliverable-oriented metrics.
    """

    @abstractmethod
    def evaluate(self, scm: Any, data: Any) -> float:
        """
        Evaluate this deliverable metric.

        Args:
            scm: Source control management interface or data.
            data: Additional deliverable-related data.

        Returns:
            float: The deliverable metric score.
        """
