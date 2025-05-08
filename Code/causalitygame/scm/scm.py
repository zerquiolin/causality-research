# Math
import numpy as np

# Graph
import networkx as nx

# DAG
from .dag import DAG

# Nodes
from causalitygame.scm import EquationBasedNumericalSCMNode, EquationBasedCategoricalSCMNode
from causalitygame.scm.base import ACCESSIBILITY_CONTROLLABLE, ACCESSIBILITY_LATENT, ACCESSIBILITY_OBSERVABLE

# Typing
from typing import List, Dict, Optional


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
        dag: DAG,
        nodes: List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode],
        random_state: np.random.RandomState,
        name=None
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
        self.random_state = random_state
        self.name = name
    
    @property
    def vars(self):
        return sorted(self.nodes.keys())
    
    @property
    def controllable_vars(self):
        return [n.name for n in self.nodes.values() if n.accessibility == ACCESSIBILITY_CONTROLLABLE]
    
    @property
    def observable_vars(self):
        return [n.name for n in self.nodes.values() if n.accessibility in [ACCESSIBILITY_OBSERVABLE, ACCESSIBILITY_CONTROLLABLE]]
    
    @property
    def latent_vars(self):
        return [n.name for n in self.nodes.values() if n.accessibility == ACCESSIBILITY_LATENT]

    def get_random_state(self) -> np.random.RandomState:
        """
        Returns the SCM's random number generator.

        Returns:
            np.random.RandomState: The random generator used for sampling.
        """
        return self.random_state

    def _generate_sample(
        self,
        interventions: Dict[str, float] = {},
        random_state: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """
        Generates a single sample from the SCM.

        Args:
            interventions (Dict[str, float], optional): A dictionary of variable names and values to intervene on.
            random_state (np.random.RandomState, optional): A custom random generator. Defaults to self.random_state.

        Returns:
            Dict[str, float]: A dictionary mapping variable names to sampled values.
        """
        rs = random_state or self.random_state
        sample = {}

        for node_name, node in [
            (node_name, self.nodes[node_name])
            for node_name in list(nx.topological_sort(self.dag.graph))
        ]:
            if node_name in interventions:
                sample[node_name] = interventions[node_name]
            else:
                value = node.generate_value(sample, random_state=rs)
                sample[node_name] = value

        return sample

    def generate_samples(
        self,
        interventions: Dict[str, float] = {},
        num_samples: int = 1,
        random_state: Optional[np.random.RandomState] = None,
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
        rs = random_state or self.random_state
        return [
            self._generate_sample(interventions, random_state=rs)
            for _ in range(num_samples)
        ]

    def to_dict(self) -> Dict:
        """
        Serializes the SCM to a dictionary format.

        Returns:
            Dict: A dictionary representing the SCM's structure and state.
        """
        # Serialize nodes and their parameters
        nodes_data = [node.to_dict() for node in self.nodes.values()]

        # Serialize the random state
        state = self.random_state.get_state()
        state_dict = {
            "state": state[0],
            "keys": state[1].tolist(),
            "pos": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        }

        return {
            "vars": nodes_data,
            "edges": self.dag.to_dict()["edges"],
            "random_state": state_dict,
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
        # Reconstruct the DAG from the dictionary
        nodes = [v["name"] for v in data["vars"]]
        edges = data["edges"]
        dag = DAG.from_dict({
            "nodes": nodes,
            "edges": edges
        })

        # Ensure nodes are sorted in topological order
        topological_order = list(nx.topological_sort(dag.graph))
        nodes = []

        # Create nodes in topological order
        for node_as_dict in sorted(data["vars"], key=lambda n: topological_order.index(n["name"])):
            if node_as_dict["class"] == EquationBasedNumericalSCMNode.__name__:
                node = EquationBasedNumericalSCMNode.from_dict(node_as_dict)
            elif node_as_dict["class"] == EquationBasedCategoricalSCMNode.__name__:
                node = EquationBasedCategoricalSCMNode.from_dict(node_as_dict)
            else:
                raise ValueError(f"Unknown node class: {node_as_dict['class']}")
            nodes.append(node)

        # Reconstruct the random state
        random_state = np.random.RandomState()
        if "random_state" in data:
            state_tuple = (
                str(data["random_state"]["state"]),
                np.array(data["random_state"]["keys"], dtype=np.uint32),
                int(data["random_state"]["pos"]),
                int(data["random_state"]["has_gauss"]),
                float(data["random_state"]["cached_gaussian"]),
            )
            random_state.set_state(state_tuple)

        return cls(dag, nodes, random_state)
