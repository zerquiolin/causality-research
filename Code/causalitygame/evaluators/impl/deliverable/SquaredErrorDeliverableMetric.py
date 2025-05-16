from ...base import DeliverableMetric
from typing import Tuple


class SquaredErrorDeliverableMetric(DeliverableMetric):
    name = "Squared Error Deliverable Metric"

    def evaluate(self, scm, data: Tuple[float, float]) -> float:
        return (data[0] - data[1]) ** 2
