# Math
import numpy as np

import pandas as pd

import logging

# Graph
import networkx as nx

from causalitygame.lib.utils.random_state_serialization import (
    random_state_from_json,
    random_state_to_json,
)

# Nodes
from causalitygame.scm.node.base import (
    ACCESSIBILITY_CONTROLLABLE,
    ACCESSIBILITY_LATENT,
    ACCESSIBILITY_OBSERVABLE,
    BaseSCMNode,
)

# Typing
from typing import Any, Callable, Dict, List, Optional, Tuple
from causalitygame.lib.utils.imports import get_class

# Abstract
from abc import ABC, abstractmethod


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


class SCM:
    """
    Structural Causal Model (SCM) that represents a system of variables with causal dependencies.

    It generates data samples according to a DAG and a collection of SCMNodes, where each node defines
    a data generation function. The nodes are evaluated in topological order of the DAG, respecting
    causal dependencies.

    Attributes:
        dag (DAG): The underlying directed acyclic graph defining variable dependencies.
        nodes (List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode]): List of nodes in topological order.
        random_state (np.random.RandomState): Random number generator for reproducibility.
    """

    def __init__(
        self,
        dag: BaseDAG,
        nodes: List[BaseSCMNode],
        random_state: Optional[np.random.RandomState],
        logger: logging.Logger = None,
        name=None,
    ):
        """
        Initializes the SCM with a DAG, a list of nodes, and a random number generator.

        Args:
            dag (DAG): The DAG representing the causal structure.
            nodes (List[SCMNode]): List of SCMNode instances in topological order.
            random_state (np.random.RandomState): NumPy random number generator.
            name (str): Name of the SCM
        """
        self.dag = dag
        self.nodes = {node.name: node for node in nodes}
        self._topologically_sorted_var_names = list(nx.topological_sort(self.dag.graph))
        self.random_state = random_state if random_state else np.random.RandomState(911)
        self.name = name
        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        )

    @property
    def vars(self):
        return self._topologically_sorted_var_names

    @property
    def controllable_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility == ACCESSIBILITY_CONTROLLABLE
        ]

    @property
    def observable_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility
            in [ACCESSIBILITY_CONTROLLABLE, ACCESSIBILITY_OBSERVABLE]
        ]

    @property
    def latent_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility in [ACCESSIBILITY_LATENT]
        ]

    @property
    def outcome_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility
            in [ACCESSIBILITY_CONTROLLABLE, ACCESSIBILITY_OBSERVABLE]
            and not self.nodes[n].parents
        ]

    def get_random_state(self) -> np.random.RandomState:
        """
        Returns the SCM's random number generator.

        Returns:
            np.random.RandomState: The random generator used for sampling.
        """
        return self.random_state

    def prepare_new_random_state_structure(self, random_state=None):

        # root
        random_state = random_state or self.random_state

        # ask each node for a structure of new random states
        random_structure = {}
        for node_name, node in self.nodes.items():
            random_structure[node_name] = node.prepare_new_random_state_structure(
                random_state
            )
        return random_structure

    def generate_samples(
        self,
        interventions: Dict[str, float] = {},
        num_samples: int = 1,
        random_state: Optional[Dict[str, np.random.RandomState]] = None,
    ) -> List[Dict[str, float]]:
        """
        Generates multiple samples from the SCM.

        Args:
            interventions (Dict[str, float], optional): Interventions to apply to nodes.
            num_samples (int): Number of samples to generate.
            random_state (np.random.RandomState, optional): Optional random generator for reproducibility.

        Returns:
            List[Dict[str, float]]: A list of sample dictionaries.
        """
        # print("Interventions:", interventions)
        random_states = random_state or {v: self.random_state for v in self.vars}
        sample = pd.DataFrame(index=range(num_samples))

        for node_name, node in [
            (node_name, self.nodes[node_name]) for node_name in self.vars
        ]:
            if node_name in interventions:
                sample_for_col = [interventions[node_name]] * num_samples
            else:
                sample_for_col = node.generate_values(
                    parent_values=sample,
                    random_state=(
                        random_states[node_name]
                        if isinstance(random_states, dict)
                        else random_states
                    ),
                )
            sample = pd.concat(
                [sample, pd.DataFrame({node_name: sample_for_col})], axis=1
            )
        assert (
            type(sample) == pd.DataFrame
        ), f"sample should be a dataframe but is {type(sample)}"
        return sample

    def to_dict(self) -> Dict:
        """
        Serializes the SCM to a dictionary format.

        Returns:
            Dict: A dictionary representing the SCM's structure and state.
        """
        # Serialize nodes and their parameters
        nodes_data = [node.to_dict() for node in self.nodes.values()]

        # Serialize the random state
        return {
            "vars": nodes_data,
            "edges": self.dag.to_dict()["edges"],
            "random_state": random_state_to_json(self.random_state),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SCM":
        """
        Deserializes an SCM instance from a dictionary.

        Args:
            data (Dict): Dictionary containing SCM structure and state.

        Returns:
            SCM: A new SCM instance.
        """
        if "class" in data and data["class"] not in [__class__.__name__]:
            class_name = data.pop("class")
            return get_class(class_name).from_dict(data)

        from causalitygame.scm.dag import DAG  # TODO

        # Reconstruct the DAG from the dictionary
        nodes = [v["name"] for v in data["vars"]]
        edges = data["edges"]
        dag = DAG.from_dict({"nodes": nodes, "edges": edges})

        # Ensure nodes are sorted in topological order
        topological_order = list(nx.topological_sort(dag.graph))
        nodes = []

        # Create nodes in topological order
        for node_as_dict in sorted(
            data["vars"], key=lambda n: topological_order.index(n["name"])
        ):
            # extract parents from edges if they are not explicitly given
            if not "parents" in node_as_dict:
                node_as_dict["parents"] = [
                    e[0] for e in edges if e[1] == node_as_dict["name"]
                ]

            # Generate parent mappings if not provided
            if (
                not "parent_mappings" in node_as_dict
                or not node_as_dict["parent_mappings"]
            ):
                node_as_dict["parent_mappings"] = {
                    node.name: {cat: idx for idx, cat in enumerate(node.domain)}
                    for node in nodes
                    if node.name in node_as_dict["parents"]
                    and isinstance(node.domain[0], str)
                }

            # get node object
            nodes.append(BaseSCMNode.from_dict(node_as_dict))

        # Reconstruct the random state
        random_state = (
            random_state_from_json(data["random_state"])
            if "random_state" in data
            else None
        )
        return cls(dag, nodes, random_state)
