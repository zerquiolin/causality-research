# Abstract
from causalitygame.scm.node.base import BaseNoiseDistribution

import numpy as np

# Distributions
from scipy.stats import norm, uniform

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


class GaussianNoiseDistribution(BaseNoiseDistribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initializes the Gaussian noise distribution with a given mean and standard deviation.

        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
        """
        self.mean = mean
        self.std = std

    def generate(self, size, random_state: int = 911) -> float:
        """
        Generates a noise value using the Gaussian distribution.

        Args:
            random_state (int, optional): Seed for random number generation. Defaults to 911.

        Returns:
            float: A generated noise value.
        """
        return norm.rvs(loc=self.mean, scale=self.std, size=size, random_state=random_state)

    def to_dict(self) -> dict:
        """
        Serializes the Gaussian noise object into a dictionary format.

        Returns:
            dict: The dictionary representation of the Gaussian noise object.
        """
        return {
            "class": GaussianNoiseDistribution.__name__,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GaussianNoiseDistribution":
        """
        Deserializes the Gaussian noise object from a dictionary representation.

        Args:
            data (dict): The dictionary containing Gaussian noise data.

        Returns:
            GaussianNoiseDistribution: An instance of Gaussian noise distribution.
        """
        return cls(mean=data["mean"], std=data["std"])


class UniformNoiseDistribution(BaseNoiseDistribution):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        """
        Initializes the Uniform noise distribution with a given range.

        Args:
            low (float): The lower bound of the uniform distribution.
            high (float): The upper bound of the uniform distribution.
        """
        self.low = low
        self.high = high

    def generate(self, size, random_state: int = 911) -> float:
        """
        Generates a noise value using the Uniform distribution.

        Args:
            random_state (int, optional): Seed for random number generation. Defaults to 911.

        Returns:
            float: A generated noise value.
        """
        return uniform.rvs(
            loc=self.low, scale=self.high - self.low, size=size, random_state=random_state
        )

    def to_dict(self) -> dict:
        """
        Serializes the Uniform noise object into a dictionary format.

        Returns:
            dict: The dictionary representation of the Uniform noise object.
        """
        return {
            "class": UniformNoiseDistribution.__name__,
            "low": self.low,
            "high": self.high,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UniformNoiseDistribution":
        """
        Deserializes the Uniform noise object from a dictionary representation.

        Args:
            data (dict): The dictionary containing Uniform noise data.

        Returns:
            UniformNoiseDistribution: An instance of Uniform noise distribution.
        """
        return cls(low=data["low"], high=data["high"])
