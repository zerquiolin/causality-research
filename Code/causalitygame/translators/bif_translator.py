from .base import BaseBayesianNetworkTranslator, BayesianNetworkGraph
from pgmpy.readwrite import BIFReader
from itertools import product


class BifTranslator(BaseBayesianNetworkTranslator):
    def translate(self, file_path: str) -> BayesianNetworkGraph:
        reader = BIFReader(file_path)
        model = reader.get_model()

        variables = {}

        for node in model.nodes():
            cpd = model.get_cpds(node)
            states = reader.get_states()[node]
            parents = list(cpd.get_evidence()) if cpd.get_evidence() else []
            cpd_values = cpd.get_values()

            if not parents:
                prob = cpd_values.tolist()
            else:
                prob = {}
                parent_states = [reader.get_states()[p] for p in parents]

                for index, parent_values in enumerate(product(*parent_states)):
                    key = ",".join(parent_values)
                    prob[key] = cpd_values[:, index].tolist()

            variables[node] = {
                "values": states,
                "parents": parents,
                "probability_distribution": prob,
            }

        edges = [list(edge) for edge in model.edges()]
        return BayesianNetworkGraph(
            nodes=variables, edges=edges, distributions=variables
        )
