# Abstract
from .abstract import BaseNoiseDistribution

# Distributions
from scipy.stats import uniform


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
            loc=self.low,
            scale=self.high - self.low,
            size=size,
            random_state=random_state,
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
