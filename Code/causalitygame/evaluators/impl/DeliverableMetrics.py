import numpy as np
import networkx as nx
from ..base import DeliverableMetric
from typing import List, Any


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

        return 1 - len(diff)


class F1DeliverableMetric(DeliverableMetric):
    name = "F1"

    def evaluate(self, scm, history) -> float:
        G_true: nx.DiGraph = scm.dag.graph
        G_pred: nx.DiGraph = history.iloc[-1]["action_object"]
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


class EdgeAccuracyDeliverableMetric(DeliverableMetric):
    name = "EdgeOrientationAccuracy"

    def evaluate(self, scm, history) -> float:
        G_true: nx.DiGraph = scm.dag.graph
        G_pred: nx.DiGraph = history.iloc[-1]["action_object"]

        true_edges = set(G_true.edges())
        pred_edges = set(G_pred.edges())
        correct = sum((u, v) in true_edges for (u, v) in pred_edges)

        return correct / len(pred_edges) if pred_edges else 0.0


class PEHEDeliverableMetric(DeliverableMetric):
    name = "PEHE"

    def __init__(self, true_effects: np.ndarray, predicted_effects: np.ndarray):
        self.true_effects = true_effects
        self.predicted_effects = predicted_effects

    def evaluate(self, history) -> float:
        errors = (self.true_effects - self.predicted_effects) ** 2
        return float(np.sqrt(np.mean(errors)))


class PolicyRiskDeliverableMetric(DeliverableMetric):
    name = "PolicyRisk"

    def __init__(self, policy_fn, contexts: List[Any], outcomes: List[Any]):
        self.policy_fn = policy_fn
        self.contexts = contexts
        self.outcomes = outcomes

    def evaluate(self, history) -> float:
        regrets = []
        for ctx, actual in zip(self.contexts, self.outcomes):
            chosen = self.policy_fn(ctx)
            optimal = actual  # assume outcome indicates optimal action
            regrets.append(0 if chosen == optimal else 1)
        return float(np.mean(regrets))
