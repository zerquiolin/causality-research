# Abstract
from ..abstract import DeliverableMetric

# Science
import numpy as np

# Types
from causalitygame.scm.abstract import SCM


class PEHEDeliverableMetric(DeliverableMetric):
    name = "PEHE Deliverable Metric"

    def __init__(self, true_effects: np.ndarray, predicted_effects: np.ndarray):
        self.true_effects = true_effects
        self.predicted_effects = predicted_effects

    def mount(self, scm: SCM) -> None:  # unused
        pass

    def evaluate(self, history) -> float:
        errors = (self.true_effects - self.predicted_effects) ** 2
        return float(np.sqrt(np.mean(errors)))
