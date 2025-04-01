# Math
import numpy as np

# Graph
import networkx as nx

# DAG
from .dag import DAG

# Nodes
from .nodes import SCMNode

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
        nodes (Dict[str, SCMNode]): Dictionary of variable nodes by name.
        random_state (np.random.RandomState): Random number generator for reproducibility.
    """

    def __init__(
        self, dag: DAG, nodes: List[SCMNode], random_state: np.random.RandomState
    ):
        """
        Initializes the SCM with a DAG, a list of nodes, and a random number generator.

        Args:
            dag (DAG): The DAG representing the causal structure.
            nodes (List[SCMNode]): List of SCMNode instances in topological order.
            random_state (np.random.RandomState): NumPy random number generator.
        """
        self.dag = dag
        self.nodes = {node.name: node for node in nodes}
        self.random_state = random_state

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
                if node.var_type == "categorical":
                    sample[node_name + "_num"] = node.input_numeric
            else:
                value = node.generate_value(sample, random_state=rs)
                sample[node_name] = value
                if node.var_type == "categorical":
                    sample[node_name + "_num"] = node.input_numeric

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
        nodes_data = {name: node.to_dict() for name, node in self.nodes.items()}
        state = self.random_state.get_state()

        state_dict = {
            "state": state[0],
            "keys": state[1].tolist(),
            "pos": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        }

        return {
            "nodes": nodes_data,
            "dag": self.dag.to_dict(),
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
        nodes = [SCMNode.from_dict(nd) for nd in data["nodes"].values()]
        nodes.sort(
            key=lambda n: int(n.name[1:])
        )  # Ensure topological order if node names follow 'X1', 'X2', ...

        dag = DAG.from_dict(data["dag"])

        state_tuple = (
            str(data["random_state"]["state"]),
            np.array(data["random_state"]["keys"], dtype=np.uint32),
            int(data["random_state"]["pos"]),
            int(data["random_state"]["has_gauss"]),
            float(data["random_state"]["cached_gaussian"]),
        )

        random_state = np.random.RandomState()
        random_state.set_state(state_tuple)

        return cls(dag, nodes, random_state)
