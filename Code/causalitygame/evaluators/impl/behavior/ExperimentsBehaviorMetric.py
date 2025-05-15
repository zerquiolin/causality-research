from ...base import BehaviorMetric
from causalitygame.lib.utils.metrics import log_penalty


class ExperimentsBehaviorMetric(BehaviorMetric):
    name = "experiments_score"

    def __init__(self, e: float = 1.0):
        self.e = e

    def evaluate(self, history) -> float:
        nE = sum(
            len(row["action_object"])
            for _, row in history.iloc[:-1].iterrows()
            if row["action"] != "stop_with_answer"
        )
        return log_penalty(nE, alpha=0.30)
