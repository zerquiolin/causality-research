import numpy as np
from causalitygame.translators.base import BaseBayesianNetworkTranslator
from causalitygame.scm.node.nodes import BayesianNetworkSCMNode
import networkx as nx
from causalitygame.scm.dag import DAG
from causalitygame.scm.scm import SCM


class BayesianNetworkBasedSCMGenerator:
    def __init__(self, translator: BaseBayesianNetworkTranslator, file_path: str):
        """
        Initialize the generator with a translator and a file path.

        Args:
            translator (BayesianNetworkTranslator): A concrete implementation like BifTranslator.
            file_path (str): Path to the BIF or other network file.
        """
        self.translator = translator
        self.graph = translator.translate(file_path)

    def _generate_nodes(self):
        """
        Create a list of BayesianNetworkSCMNode instances from the graph.

        Returns:
            List[BayesianNetworkSCMNode]: SCM node representations of the variables.
        """
        nodes = []
        for name, data in self.graph.nodes.items():
            node = BayesianNetworkSCMNode(
                name=name,
                parents=data["parents"],
                values=data["values"],
                probability_distribution=data["probability_distribution"],
            )
            nodes.append(node)
        return nodes

    def generate(self) -> SCM:
        """
        Generate a Structural Causal Model (SCM) from the Bayesian network.

        Returns:
            SCM: The generated SCM.
        """
        nodes = self._generate_nodes()
        graph = nx.DiGraph()
        graph.add_edges_from(self.graph.edges)
        dag = DAG(graph)

        random_state = np.random.RandomState(42)

        return SCM(dag=dag, nodes=nodes, random_state=random_state)
