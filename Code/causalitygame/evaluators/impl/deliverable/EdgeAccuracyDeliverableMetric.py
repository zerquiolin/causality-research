import numpy as np
import networkx as nx
from ...base import DeliverableMetric


class EdgeAccuracyDeliverableMetric(DeliverableMetric):
    name = "EdgeOrientationAccuracy"

    def evaluate(self, scm, history) -> float:
        G_true: nx.DiGraph = scm.dag.graph
        G_pred: nx.DiGraph = history.iloc[-1]["action_object"]

        true_edges = set(G_true.edges())
        pred_edges = set(G_pred.edges())
        correct = sum((u, v) in true_edges for (u, v) in pred_edges)

        return correct / len(pred_edges) if pred_edges else 0.0
