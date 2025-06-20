# Abstract
from .abstract import BaseMission

# Utils
from causalitygame.lib.utils.imports import find_importable_classes

# Types
from causalitygame.evaluators.abstract import (
    BehaviorMetric,
    DeliverableMetric,
)

# Constants
from causalitygame.lib.constants.routes import METRICS_FOLDER_PATH

# Identify specific metric classes
metric_classes = find_importable_classes(METRICS_FOLDER_PATH, base_class=BehaviorMetric)
metric_classes.update(
    find_importable_classes(METRICS_FOLDER_PATH, base_class=DeliverableMetric)
)


class DAGInferenceMission(BaseMission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    underlying causal structure from observational data.
    """

    name = "DAG Inference Mission"
    description = "This mission evaluates the ability to infer the structure of a DAG from observational data. The user is expected to provide a nx.DiGraph object representing the underlying causal structure."

    def __init__(self, behavior_metric, deliverable_metric):
        super().__init__(behavior_metric, deliverable_metric)

    def evaluate(self, scm, history):
        behavior_score = self.behavior_metric.evaluate(history)
        deliverable_score = self.deliverable_metric.evaluate(scm=scm, history=history)
        return behavior_score, deliverable_score

    def to_dict(self):
        return {
            "class": DAGInferenceMission.__name__,
            "behavior_metric": self.behavior_metric.__class__.__name__,
            "deliverable_metric": self.deliverable_metric.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, dict):
        global metric_classes
        # Check if the behavior metric class is known
        behavior_cls = metric_classes.get(dict["behavior_metric"])
        if behavior_cls is None:
            raise ValueError(f"Unknown mission class: {dict['behavior_metric']}")
        # Instantiate the mission from the data
        behavior_metric = behavior_cls()
        # Check if the deliverable metric class is known
        deliverable_cls = metric_classes.get(dict["deliverable_metric"])
        if deliverable_cls is None:
            raise ValueError(f"Unknown mission class: {dict['deliverable_metric']}")
        # Instantiate the mission from the data
        deliverable_metric = deliverable_cls()

        return cls(
            behavior_metric=behavior_metric,
            deliverable_metric=deliverable_metric,
        )
