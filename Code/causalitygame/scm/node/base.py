from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

ACCESSIBILITY_LATENT = "latent"
ACCESSIBILITY_OBSERVABLE = "observable"
ACCESSIBILITY_CONTROLLABLE = "controllable"

from causalitygame.lib.utils.random_state_serialization import (
    random_state_from_json,
    random_state_to_json,
)
from causalitygame.lib.utils.imports import get_class

import numpy as np
import pandas as pd

import logging


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


class BaseSCMNode(ABC):
    def __init__(
        self,
        name: str,
        evaluation: Optional[Callable],
        domain: List[float | str],
        noise_distribution: BaseNoiseDistribution,
        accessibility: str = ACCESSIBILITY_CONTROLLABLE,
        parents: Optional[List[str]] = None,
        parent_mappings: Optional[Dict[str, int | float]] = None,
        random_state: np.random.RandomState = np.random.RandomState(911),
        logger: logging.Logger = None,
    ):
        """
        SCMNode is class representing a node in a Structural Causal Model (SCM).
        It encapsulates the node's name, evaluation function, domain of possible values,
        parent nodes, and a random state for generating random values.

        Args:
            name (str): The name of the node.
            accessibility (str): accessibility of this variable by the agent (latent, observable, or controllable)
            evaluation (Callable): A function to evaluate the node's value based on its parents.
            domain (List[float | str]): The domain of possible values for the node.
            parents (List[str]): A list of parent node names.
            random_state (np.random.RandomState): Random state for generating random values.
        """
        self.name = name
        self.accessibility = accessibility
        self.evaluation = evaluation
        self.domain = domain
        if not isinstance(domain, list):
            self.domain = list(self.domain)
        self.noise_distribution = noise_distribution
        self.parents = parents
        self.parent_mappings = parent_mappings
        self.random_state = random_state
        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        )

        # this is just to not break the MRO
        super().__init__()

    def _init_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

    def prepare_new_random_state_structure(self, random_state):
        """
            generates a random structure that is required by this node. By default, this is just a simple RandomState.
            However, if need be and keeping in mind reproducibility, it can be useful to generate several such objects
            so that several random things can be determined for multiple sampled instances in parallel, e.g., noise and category or so.

        Args:
            random_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.random.RandomState(random_state.randint(0, 10**5))

    @abstractmethod
    def generate_values(
        self, parent_values: pd.DataFrame, random_state: np.random.RandomState
    ) -> float | str:
        """
        Generates a value for the node based on its parents and noise.

        Args:
            parent_values (dict): A dictionary of parent node values.
            random_state (np.random.RandomState): Random state for generating random values.

        Returns:
            float | str: The generated value for the node.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _to_dict(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        d = {
            "class": f"{self.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "accessibility": self.accessibility,
            "domain": self.domain,
            "parents": self.parents,
            "parent_mappings": self.parent_mappings,
            "random_state": (
                random_state_to_json(self.random_state)
                if self.random_state is not None
                else None
            ),
        }
        d.update(self._to_dict())
        assert "class" in d
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BaseSCMNode":
        assert "class" in data, f"Serialized node has no class entry: {data}"
        data = data.copy()
        data["random_state"] = (
            random_state_from_json(data["random_state"])
            if "random_state" in data and data["random_state"] is not None
            else None
        )
        class_name = data.pop("class")
        fully_qualified_class_name = (
            class_name if "." in class_name else f"causalitygame.scm.node.{class_name}"
        )
        return get_class(fully_qualified_class_name).from_dict(data)


class BaseNumericSCMNode(BaseSCMNode):
    pass


class BaseCategoricSCMNode(BaseSCMNode):
    pass
