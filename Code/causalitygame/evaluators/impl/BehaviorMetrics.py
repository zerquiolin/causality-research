from ..base import BehaviorMetric


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
        return self.e / (1 + nE**2)


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
        return self.t / (1 + nT**2)


class RoundsBehaviorMetric(BehaviorMetric):
    name = "rounds_score"

    def __init__(self, r: float = 0.2):
        self.r = r

    def evaluate(self, history) -> float:
        nR = sum(
            1
            for _, row in history.iloc[:-1].iterrows()
            if row["action"] != "stop_with_answer"
        )
        return self.r / (1 + nR**2)
