# Abstract
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# DAG
import networkx as nx


class BaseDAG(ABC):
    """
    BaseDAG is an abstract base class for Directed Acyclic Graph (DAG) structures using NetworkX.
    It provides default implementations for common DAG operations such as accessing nodes and edges,
    retrieving parent nodes, and classifying nodes as roots, leaves, or intermediates.
    Subclasses must implement serialization and visualization methods.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    @property
    def nodes(self) -> List[Any]:
        """
        Returns a list of nodes in the DAG.

        Returns:
            List[Any]: A list of node identifiers.
        """
        return list(self.graph.nodes())

    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        """
        Returns a list of edges in the DAG.

        Returns:
            List[Tuple[Any, Any]]: A list of (source, target) edge tuples.
        """
        return list(self.graph.edges())

    def get_parents(self, node: Any) -> List[Any]:
        """
        Retrieves the parent nodes (predecessors) of the given node.

        Args:
            node (Any): The node whose parents are to be retrieved.

        Returns:
            List[Any]: A list of parent nodes.
        """
        return list(self.graph.predecessors(node))

    def get_node_types(self) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Categorizes nodes in the DAG as roots (no incoming edges),
        leaves (no outgoing edges), or intermediates (both in and out).

        Returns:
            Tuple[List[Any], List[Any], List[Any]]: Lists of roots, leaves, and intermediates.
        """
        roots, leaves, intermediates = [], [], []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                roots.append(node)
            elif self.graph.out_degree(node) == 0:
                leaves.append(node)
            else:
                intermediates.append(node)
        return roots, leaves, intermediates

    def get_structured_nodes(self) -> Dict[Any, List[Any]]:
        """
        Constructs a dictionary mapping each node to its list of parent nodes.

        Returns:
            Dict[Any, List[Any]]: A mapping of each node to its parents.
        """
        return {node: self.get_parents(node) for node in self.graph.nodes()}

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the DAG into a dictionary format suitable for storage or transmission.

        Returns:
            Dict[str, Any]: The dictionary representation of the DAG.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDAG":
        """
        Deserializes the DAG from a dictionary representation.

        Args:
            data (Dict[str, Any]): The dictionary containing DAG data.

        Returns:
            BaseDAG: An instance of a DAG reconstructed from the data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def plot(self, spacing_factor: float = 2.0) -> None:
        """
        Visualizes the DAG structure.

        Args:
            spacing_factor (float, optional): A factor to adjust node spacing in the plot. Defaults to 2.0.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseSCM(ABC):
    @abstractmethod
    def get_random_state(self) -> np.random.RandomState:
        pass

    @abstractmethod
    def generate_samples(
        self,
        interventions: Dict[str, float] = {},
        num_samples: int = 1,
        random_state: Optional[np.random.RandomState] = None,
    ) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> "BaseSCM":
        pass


class BaseNoiseDistribution(ABC):
    def generate(self, random_state: Optional[int] = 911) -> float:
        """
        Generates a noise value using the provided random state.

        Args:
            random_state (int, optional): Seed for random number generation. Defaults to 911.

        Returns:
            float: A generated noise value.
        """
        return self.noise.rsv(random_state=random_state)

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
        parents: Optional[List[str]] = None,
        parent_mappings: Optional[Dict[str, int | float]] = None,
        random_state: int = 911,
    ):
        """
        SCMNode is class representing a node in a Structural Causal Model (SCM).
        It encapsulates the node's name, evaluation function, domain of possible values,
        parent nodes, and a random state for generating random values.

        Args:
            name (str): The name of the node.
            evaluation (Callable): A function to evaluate the node's value based on its parents.
            domain (List[float | str]): The domain of possible values for the node.
            parents (List[str]): A list of parent node names.
            random_state (int): Seed for random number generation.
        """
        self.name = name
        self.evaluation = evaluation
        self.domain = domain
        self.noise_distribution = noise_distribution
        self.parents = parents
        self.parent_mappings = parent_mappings
        self.random_state_seed = random_state
        self.random_state = np.random.RandomState(random_state)

    @abstractmethod
    def generate_value(self, parent_values: dict, random_state: int) -> float | str:
        """
        Generates a value for the node based on its parents and noise.

        Args:
            parent_values (dict): A dictionary of parent node values.
            random_state (int): Seed for random number generation.

        Returns:
            float | str: The generated value for the node.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseSCMNode":
        raise NotImplementedError("Subclasses must implement this method.")
