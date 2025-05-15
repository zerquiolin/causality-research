import networkx as nx
from ...base import DeliverableMetric


class SHDDeliverableMetric(DeliverableMetric):
    name = "SHD"

    def evaluate(self, scm, history) -> float:
        G_true: nx.DiGraph = scm.dag.graph
        G_pred: nx.DiGraph = history.iloc[-1]["action_object"]

        nodes = sorted(set(G_true.nodes()).union(G_pred.nodes()))
        G_true = nx.relabel_nodes(G_true.subgraph(nodes), lambda x: x)
        G_pred = nx.relabel_nodes(G_pred.subgraph(nodes), lambda x: x)

        true_edges = set(G_true.edges())
        pred_edges = set(G_pred.edges())
        und_true = set(map(frozenset, true_edges))
        und_pred = set(map(frozenset, pred_edges))
        diff = und_true.symmetric_difference(und_pred)

        return len(diff)
