# Absrtract
from .base import BaseNoiseDistribution

# Math
import numpy as np


class DiracNoiseDistribution(BaseNoiseDistribution):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def generate(self, size, random_state) -> float:
        """
        Always provides the same value

        Returns:
            float: the constant value
        """
        return self.val * np.ones(size)

    def to_dict(self):
        """
        Serializes the noise object into a dictionary format.

        Returns:
            Dict: The dictionary representation of the noise object.
        """
        return {"val": self.val}

    def from_dict(cls, data):
        return DiracNoiseDistribution(**data)
