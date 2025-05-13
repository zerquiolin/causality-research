from abc import ABC, abstractmethod
from causalitygame.scm.base import get_class
from causalitygame.lib.utils.random_state_serialization import random_state_to_json, random_state_from_json
import numpy as np


class OutcomeGenerator(ABC):

    def __init__(
            self,
            random_state
        ):
        super().__init__()

        # save these variables
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
            "random_state": random_state_to_json(self.random_state),
            "class": f"{self.__module__}.{self.__class__.__name__}"
        }
        d.update(self._to_dict())
        return d
    
    @classmethod
    def from_dict(cls, data):
        act_cls = data.pop("class")
        data["random_state"] = random_state_from_json(data["random_state"])
        return get_class(act_cls)._from_dict(data)


class DummyOutcomeGenerator(OutcomeGenerator):
    
    def __init__(
            self,
            constant=0
        ):
        super().__init__(random_state=None)
        self.constant = constant
    
    def fit(self, x, t, y):
        pass

    def generate(self, x, t):
        t = t.reshape(-1)
        return np.ones(len(t)) * self.constant

    def _to_dict(self):
        return {
            "constant": self.constant
        }
    
    @classmethod
    def _from_dict(cls, data):
        return DummyOutcomeGenerator(**data)


class ComplementaryOutcomeGenerator(OutcomeGenerator):
    """
        Only generates an outcome if none is given in the data

    Args:
        OutcomeGenerator (_type_): _description_
    """

    def __init__(self, base_outcome_generator, random_state=None):
        super().__init__(random_state=random_state)
        self.base_outcome_generator = base_outcome_generator

        # state vars
        self.x = None
        self.t = None
        self.y = None
    
    def fit(self, x, t, y):
        self.x = x
        self.t = t
        self.y = y
        self.base_outcome_generator.fit(x, t, y)
    
    def generate(self, x, t):
        outcomes = []
        for _x, _t in zip(x, t):
            match = np.all(self.x == _x, axis=1) & np.all(self.t == _t, axis=1)
            indices_of_match = np.where(match)[0]
            if len(indices_of_match) > 0:
                outcomes.append(self.y[self.random_state.choice(indices_of_match)])
            else:
                outcomes.append(self.base_outcome_generator.generate(np.array([_x]), np.array([_t]))[0])
        return np.array(outcomes)

    def _to_dict(self):
        return {
            "x": self.x.tolist() if self.x is not None else None,
            "t": self.t.tolist() if self.x is not None else None,
            "y": self.y.tolist() if self.x is not None else None,
            "base_generator": self.base_outcome_generator.to_dict()
        }
    
    @classmethod
    def _from_dict(cls, data):
        gen = ComplementaryOutcomeGenerator(
            base_outcome_generator=OutcomeGenerator.from_dict(data["base_generator"])
        )
        print(data)
        gen.x = np.array(data["x"])
        gen.t = np.array(data["t"])
        gen.y = np.array(data["y"])
        return gen
