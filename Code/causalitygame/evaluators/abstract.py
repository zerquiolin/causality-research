# Abstract
from abc import ABC, abstractmethod

# Types
from typing import Any
from causalitygame.scm.abstract import SCM


class BaseMetric(ABC):
    """
    Abstract base class providing a common interface for all metrics.
    """

    @abstractmethod
    def mount(self, scm: SCM) -> None:
        """
        Prepare the current metric for evaluation.

        Args:
            scm (SCM): Structural Causal Model.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> float:
        """
        Evaluate the metric.

        Returns:
            float: The metric result.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


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
        raise NotImplementedError("This method should be implemented by subclasses.")


class DeliverableMetric(BaseMetric):
    """
    Base class for deliverable-oriented metrics.
    """

    @abstractmethod
    def evaluate(self, scm: SCM, data: Any) -> float:
        """
        Evaluate this deliverable metric.

        Args:
            scm: Structural Causal Model.
            data: Additional deliverable-related data.

        Returns:
            float: The deliverable metric score.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
