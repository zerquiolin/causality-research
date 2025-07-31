# Abstract
from ..abstract import DeliverableMetric

# Science
import numpy as np

# Types
from typing import List, Any
from causalitygame.scm.abstract import SCM


class PolicyRiskDeliverableMetric(DeliverableMetric):
    name = "Policy Risk Deliverable Metric"

    def __init__(self, policy_fn, contexts: List[Any], outcomes: List[Any]):
        self.policy_fn = policy_fn
        self.contexts = contexts
        self.outcomes = outcomes

    def mount(self, scm: SCM) -> None:  # unused
        pass

    def evaluate(self, history) -> float:
        regrets = []
        for ctx, actual in zip(self.contexts, self.outcomes):
            chosen = self.policy_fn(ctx)
            optimal = actual  # assume outcome indicates optimal action
            regrets.append(0 if chosen == optimal else 1)
        return float(np.mean(regrets))
