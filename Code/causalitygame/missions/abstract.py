# Abstract
from abc import ABC, abstractmethod
from causalitygame.evaluators.abstract import BehaviorMetric, DeliverableMetric

# Types
from causalitygame.scm.abstract import SCM


class BaseMission(ABC):
    """
    Base class for defining missions in a game.
    This class provides an interface for evaluating the mission's performance
    and generating deliverables based on the mission's objectives.
    """

    # Description
    name: str
    description: str

    # Attributes
    is_mounted: bool = False

    def __init__(
        self, behavior_metric: BehaviorMetric, deliverable_metric: DeliverableMetric
    ):
        self.behavior_metric = behavior_metric
        self.deliverable_metric = deliverable_metric

    def mount(self, scm: SCM) -> None:
        """
        Prepare the mission to be mounted to a given SCM.

        Args:
            scm (SCM): Structural Causal Model to mount the mission to.
        """
        # Mount Behavior Metric
        self.behavior_metric.mount(scm)
        # Mount Deliverable Metric
        self.deliverable_metric.mount(scm)
        # Update the mounted state
        self.is_mounted = True

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
