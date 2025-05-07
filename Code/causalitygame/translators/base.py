from abc import ABC, abstractmethod
from typing import Dict


class BaseBayesianNetworkTranslator(ABC):
    @abstractmethod
    def translate(self, file_path: str) -> "BayesianNetworkGraph":
        """
        Translates the given file into a structured object representing
        a Bayesian network.

        Args:
            file_path (str): Path to the Bayesian network file (e.g., .bif).

        Returns:
            BayesianNetworkGraph: Structured representation of the network.
        """
        pass


class BayesianNetworkGraph:
    """
    Represents a structured Bayesian network graph.
    It contains nodes, edges, and probability distributions.

    Attributes:
        nodes (Dict): A dictionary of variable names to their data.
        edges (list): A list of directed edges (parent, child).
        distributions (Dict): Probability distribution data for each variable.

    Example:
    nodes = ['A', 'B', 'C']
    edges = [('A', 'B'), ('A', 'C')]



    """

    def __init__(self, nodes: Dict, edges: list, distributions: Dict):
        """
        Represents the structured Bayesian network graph.

        Args:
            nodes (Dict): A dictionary of variable names to their data.
            edges (list): A list of directed edges (parent, child).
            distributions (Dict): Probability distribution data for each variable.
        """
        self.nodes = nodes
        self.edges = edges
        self.distributions = distributions

    def get_node(self, name: str) -> Dict:
        """
        Retrieves data for a specific variable node.

        Args:
            name (str): The variable name.

        Returns:
            dict: Variable metadata including values, parents, and distributions.
        """
        return self.nodes[name]
