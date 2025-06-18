import numpy as np
import networkx as nx
from ..abstract import DeliverableMetric


class F1DeliverableMetric(DeliverableMetric):
    name = "F1 Deliverable Metric"

    def evaluate(self, scm, history) -> float:
        G_true: nx.DiGraph = scm.dag.graph
        G_pred: nx.DiGraph = history.iloc[-1]["current_result"]
        nodes = sorted(set(G_true.nodes()).union(G_pred.nodes()))

        true_edges = set(G_true.subgraph(nodes).edges())
        pred_edges = set(G_pred.subgraph(nodes).edges())

        y_true = [
            1 if (u, v) in true_edges else 0 for u in nodes for v in nodes if u != v
        ]
        y_pred = [
            1 if (u, v) in pred_edges else 0 for u in nodes for v in nodes if u != v
        ]

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
        fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
        fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))

        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)
