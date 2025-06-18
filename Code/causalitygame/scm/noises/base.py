# Abstract
from abc import ABC, abstractmethod

# Typing
from typing import Dict, Optional


class BaseNoiseDistribution(ABC):
    def generate(self, size, random_state: Optional[int] = 911) -> float:
        """
        Generates a noise value using the provided random state.

        Args:
            random_state (int, optional): Seed for random number generation. Defaults to 911.

        Returns:
            float: A generated noise value.
        """
        return self.noise.rsv(random_state=random_state, size=size)

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Serializes the noise object into a dictionary format.

        Returns:
            Dict: The dictionary representation of the noise object.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def from_dict(cls, data: Dict) -> "BaseNoiseDistribution":
        """
        Deserializes the noise object from a dictionary representation.

        Args:
            data (Dict): The dictionary containing noise data.

        Returns:
            BaseNoise: An instance of a noise object reconstructed from the data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
