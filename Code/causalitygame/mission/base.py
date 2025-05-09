from abc import ABC, abstractmethod
from causalitygame.evaluators.base import BehaviorMetric, DeliverableMetric


class BaseMission(ABC):
    """
    Base class for defining missions in a game.
    This class provides an interface for evaluating the mission's performance
    and generating deliverables based on the mission's objectives.
    """

    name: str

    def __init__(
        self, behavior_metric: BehaviorMetric, deliverable_metric: DeliverableMetric
    ):
        self.behavior_metric = behavior_metric
        self.deliverable_metric = deliverable_metric

    @abstractmethod
    def evaluate(self, scm, history):
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )

    @abstractmethod
    def from_dict(self, dict):
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )
