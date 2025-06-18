from ..base import BehaviorMetric
from causalitygame.lib.utils.metrics import log_penalty


class RoundsBehaviorMetric(BehaviorMetric):
    name = "Rounds Behavior Metric"

    def __init__(self, r: float = 0.2):
        self.r = r

    def evaluate(self, history) -> float:
        nR = sum(
            1
            for _, row in history.iloc[:-1].iterrows()
            if row["action"] != "stop_with_answer"
        )
        return log_penalty(nR, alpha=0.10)
