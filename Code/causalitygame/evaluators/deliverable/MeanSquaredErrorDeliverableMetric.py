from ..abstract import DeliverableMetric
from typing import List, Tuple
import numpy as np


class MeanSquaredErrorDeliverableMetric(DeliverableMetric):
    name = "Mean Squared Error Deliverable Metric"

    def evaluate(self, scm, data: Tuple[List[float], List[float]]) -> float:
        return ((np.array(data[0]) - np.array(data[1])) ** 2) / len(data[0])
