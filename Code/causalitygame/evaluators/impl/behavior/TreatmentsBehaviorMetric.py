from ...base import BehaviorMetric
from causalitygame.lib.utils.metrics import log_penalty


class TreatmentsBehaviorMetric(BehaviorMetric):
    name = "treatments_score"

    def __init__(self, t: float = 0.5):
        self.t = t

    def evaluate(self, history) -> float:
        nT = sum(
            sum(item[1] for item in row["action_object"])
            for _, row in history.iloc[:-1].iterrows()
            if row["action"] != "stop_with_answer"
        )
        return log_penalty(nT, alpha=0.20)
