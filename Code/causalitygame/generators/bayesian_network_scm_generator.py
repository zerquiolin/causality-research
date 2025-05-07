from causalitygame.translators.base import BaseBayesianNetworkTranslator
from causalitygame.scm.nodes import BayesianNetworkSCMNode


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

    def generate_nodes(self):
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
