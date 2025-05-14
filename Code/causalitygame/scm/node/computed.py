from causalitygame.scm.node.base import BaseNumericSCMNode, BaseCategoricSCMNode
from causalitygame.generators.outcome.base import OutcomeGenerator

import numpy as np

class ComputedNumericSCMNode(BaseNumericSCMNode):

    def __init__(self, name, value_computer, accessibility = ..., random_state = ...):
        super().__init__(name, None, domain=[], noise_distribution=None, accessibility=accessibility, random_state=random_state)
        self.value_computer = value_computer
    
    def generate_value(self, parent_values, random_state):
        return self.value_computer.generate(np.array([parent_values]), random_state=random_state)[0]

    def to_dict(self):
        return {
            "value_computer": self.value_computer.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        data["value_computer"] = OutcomeGenerator.from_dict(data)
        return ComputedNumericSCMNode(**data)


class ComputedCategoricSCMNode(BaseCategoricSCMNode):

    def __init__(self, name, value_computer, accessibility = ..., random_state = ...):
        super().__init__(name, None, domain=[], noise_distribution=None, accessibility=accessibility, random_state=random_state)
        self.value_computer = value_computer
    
    def generate_value(self, parent_values, random_state):
        return self.value_computer.generate(np.array([parent_values]), random_state=random_state)[0]

    def to_dict(self):
        return {
            "value_computer": self.value_computer.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        data["value_computer"] = OutcomeGenerator.from_dict(data)
        return ComputedNumericSCMNode(**data)

