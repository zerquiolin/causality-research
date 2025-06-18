import numpy as np

# import networkx as nx
from ..base import DeliverableMetric

# from typing import List, Any


class PEHEDeliverableMetric(DeliverableMetric):
    name = "PEHE Deliverable Metric"

    def __init__(self, true_effects: np.ndarray, predicted_effects: np.ndarray):
        self.true_effects = true_effects
        self.predicted_effects = predicted_effects

    def evaluate(self, history) -> float:
        errors = (self.true_effects - self.predicted_effects) ** 2
        return float(np.sqrt(np.mean(errors)))
