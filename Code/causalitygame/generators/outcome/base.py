from abc import ABC, abstractmethod
from causalitygame.scm.base import get_class
from causalitygame.lib.utils.random_state_serialization import (
    random_state_to_json,
    random_state_from_json,
)
import numpy as np

import logging


class OutcomeGenerator(ABC):

    def __init__(self, random_state, logger=None):
        super().__init__()

        # save these variables
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) in [int, np.int64]:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        )

    @abstractmethod
    def fit(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def generate(self, x, random_state=None):
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
            "class": f"{self.__module__}.{self.__class__.__name__}",
        }
        d.update(self._to_dict())
        return d

    @classmethod
    def from_dict(cls, data):
        act_cls = data.pop("class")
        data["random_state"] = random_state_from_json(data["random_state"])
        return get_class(act_cls)._from_dict(data)


class DummyOutcomeGenerator(OutcomeGenerator):

    def __init__(self, constant=0):
        super().__init__(random_state=None)
        self.constant = constant

    def fit(self, x, y):
        pass

    def generate(self, x, random_state=None):
        return np.ones(x.shape[0]) * self.constant

    def _to_dict(self):
        return {"constant": self.constant}

    @classmethod
    def _from_dict(cls, data):
        data.pop("random_state")
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
        self.y = None

    def fit(self, x, y):
        self.x = x
        if not isinstance(x, np.ndarray):
            raise ValueError(f"x must be a numpy array but is {type(x)}")
        self.y = y
        self.base_outcome_generator.fit(x, y)

    def generate(self, x: np.ndarray, random_state=None):
        if not isinstance(x, np.ndarray):
            raise ValueError(f"x must be a numpy array but is {type(x)}")

        if random_state is None:
            random_state = self.random_state

        # create matrix where entry (i, j) says whether sample i can be complemented with the value of database entry j
        matches = np.all((x[:, None, :] == self.x[None, :, :]), axis=2)
        lookupable = np.any(matches, axis=1)
        num_lookups = np.count_nonzero(lookupable)
        num_generations = len(x) - num_lookups

        # first look up all the values that can be looked up
        self.logger.info(
            f"Looking up/sampling values from existing entries for {num_lookups} samples."
        )
        lookup_values = np.array(
            [random_state.choice(self.y[match]) for match in matches[lookupable]]
        )
        assert np.count_nonzero(lookupable) == len(lookup_values)

        # now generate values for all other samples
        self.logger.info(
            f"Generating {num_generations} sample values as they are not present."
        )
        generated_values = self.base_outcome_generator.generate(x[~lookupable])
        assert (len(x) - len(lookup_values)) == len(generated_values)
        self.logger.info(f"Generated {len(generated_values)} sample values.")

        # merge the looked up and generated values position-true
        outcomes = np.zeros(len(x))
        outcomes[lookupable] = lookup_values
        outcomes[~lookupable] = generated_values
        return outcomes

    def _to_dict(self):
        return {
            "x": self.x.tolist() if self.x is not None else None,
            "y": self.y.tolist() if self.x is not None else None,
            "base_generator": self.base_outcome_generator.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data):
        gen = ComplementaryOutcomeGenerator(
            base_outcome_generator=OutcomeGenerator.from_dict(data["base_generator"])
        )
        gen.x = np.array(data["x"])
        gen.y = np.array(data["y"])
        return gen
