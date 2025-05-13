from abc import ABC, abstractmethod
from causalitygame.scm.base import get_class
from causalitygame.lib.utils.random_state_serialization import random_state_to_json, random_state_from_json
import numpy as np


class OutcomeGenerator(ABC):

    def __init__(
            self,
            outcome_variable,
            required_covariates,
            required_treatments,
            random_state
        ):
        super().__init__()

        # save these variables
        self.outcome_variable = outcome_variable
        self.required_covariates = required_covariates
        self.required_treatments = required_treatments
        
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) in [int, np.int64]:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    @abstractmethod
    def fit(self, x, t, y):
        raise NotImplementedError

    @abstractmethod
    def generate(self, x, t):
        raise NotImplementedError

    @abstractmethod
    def _to_dict(self):
        raise NotImplementedError
    
    @staticmethod
    def _from_dict(cls):
        raise NotImplementedError

    def to_dict(self):
        d = {
            "outcome_variable": self.outcome_variable,
            "required_covariates": self.required_covariates,
            "required_treatments": self.required_treatments,
            "random_state": random_state_to_json(self.random_state)
        }
        d.update(self._to_dict())
        assert "class" in d, f"improper serialization, because no class field was specified."
        return d
    
    @classmethod
    def from_dict(cls, data):
        act_cls = data.pop("class")
        data["random_state"] = random_state_from_json(data["random_state"])
        return get_class(act_cls)._from_dict(data)
