from ..base import DeliverableMetric
from typing import Tuple


class AbsoluteErrorDeliverableMetric(DeliverableMetric):
    name = "Absolute Error Deliverable Metric"

    def evaluate(self, scm, data: Tuple[float, float]) -> float:
        return abs(data[0] - data[1])
