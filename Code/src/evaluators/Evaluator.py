from src.scm.scm import SCM
from .base import DeliverableMetric, BehaviorMetric, WeightedScores
from typing import List


# === Combined Metrics Manager ===
class Evaluator:
    def __init__(
        self,
        scm: SCM,
        behavior_metrics: List[BehaviorMetric],
        deliverable_metrics: List[DeliverableMetric],
        plambda: float = 0.8,
    ):
        self.scm = scm
        self.behavior_metrics = behavior_metrics
        self.deliverable_metrics = deliverable_metrics
        self.plambda = plambda

    def evaluate(self, history):
        # Evaluate each behavior metric
        behavior_scores = [(m.name, m.evaluate(history)) for m in self.behavior_metrics]
        # Evaluate each deliverable metric
        deliverable_scores = [
            (m.name, m.evaluate(self.scm, history)) for m in self.deliverable_metrics
        ]

        # Compute weighted sums
        deliver_values = [v for _, v in deliverable_scores]
        behavior_values = [v for _, v in behavior_scores]

        weighted_deliv = sum(deliver_values) * self.plambda
        weighted_behav = sum(behavior_values) * (1 - self.plambda)

        global_score = weighted_deliv + weighted_behav

        return {
            "behavior": behavior_scores,
            "deliverable": deliverable_scores,
            "global_score": global_score,
        }
