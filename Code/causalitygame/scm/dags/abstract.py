# Abstract
from abc import ABC, abstractmethod

# Network
import networkx as nx

# Typing
from typing import Any, Dict, List, Tuple


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
