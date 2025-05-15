from causalitygame.evaluators.base import BaseMetric
from ..base import BaseMission
from causalitygame.evaluators.impl.behavior import (
    ExperimentsBehaviorMetric,
    TreatmentsBehaviorMetric,
    RoundsBehaviorMetric,
)

from causalitygame.evaluators.impl.deliverable import (
    SHDDeliverableMetric,
    F1DeliverableMetric,
    EdgeAccuracyDeliverableMetric,
)


class DAGInferenceMission(BaseMission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    underlying causal structure from observational data.
    """

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
        mapping = {
            c.__name__: c
            for c in [
                ExperimentsBehaviorMetric,
                TreatmentsBehaviorMetric,
                RoundsBehaviorMetric,
                SHDDeliverableMetric,
                F1DeliverableMetric,
                EdgeAccuracyDeliverableMetric,
            ]
        }
        behavior_metric = mapping[dict["behavior_metric"]]()
        deliverable_metric = mapping[dict["deliverable_metric"]]()

        # Ensure the behavior and deliverable metrics are initialized correctly
        assert isinstance(behavior_metric, BaseMetric), "Invalid behavior metric type"

        assert isinstance(
            deliverable_metric,
            BaseMetric,
        ), "Invalid deliverable metric type"

        return cls(
            behavior_metric=behavior_metric,
            deliverable_metric=deliverable_metric,
        )
