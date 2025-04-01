import numpy as np
from src.lib.models.scm.SCM import SCM
import networkx as nx


class CausalityMetrics:
    def __init__(self, scm: SCM, e=1.0, t=0.5, r=0.2, alpha=1.0, beta=1.0, gamma=1.0):
        self.e = e
        self.t = t
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluate_performance(self, history, goal_type):
        final_action = None
        for _, _, action, action_obj in reversed(self.history):
            if isinstance(action, str) and action.startswith("stop"):
                final_action = action_obj
                break

        if self.goal_type == "dag_discovery":
            return self._evaluate_dag(final_action)

        raise NotImplementedError(f"Goal '{self.goal_type}' not supported.")

    def _f1_score(self, y_true, y_pred, zero_division=0):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        if tp + fp == 0 or tp + fn == 0:
            return zero_division

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall == 0:
            return zero_division

        return 2 * (precision * recall) / (precision + recall)

    def _evaluate_dag(self, predicted_answer):
        G_true = self.scm.dag  # expected to be a networkx.DiGraph
        G_pred = predicted_answer  # expected to be a networkx.DiGraph

        # Ensure both graphs have the same nodes
        nodes = sorted(set(G_true.nodes()).union(set(G_pred.nodes())))
        G_true = nx.relabel_nodes(G_true.subgraph(nodes), lambda x: x)
        G_pred = nx.relabel_nodes(G_pred.subgraph(nodes), lambda x: x)

        true_edges = set(G_true.edges())
        pred_edges = set(G_pred.edges())

        # SHD: count edges that are missing, extra, or reversed
        undirected_true = set(map(frozenset, true_edges))
        undirected_pred = set(map(frozenset, pred_edges))
        undirected_diff = undirected_true.symmetric_difference(undirected_pred)
        shd = len(undirected_diff)

        # Precision, Recall, F1 for directed edges
        y_true = [
            1 if (u, v) in true_edges else 0 for u in nodes for v in nodes if u != v
        ]
        y_pred = [
            1 if (u, v) in pred_edges else 0 for u in nodes for v in nodes if u != v
        ]

        f1 = self._f1_score(y_true, y_pred, zero_division=0)

        # Edge Orientation Accuracy: correct direction / total predicted
        correct_orient = sum((u, v) in true_edges for u, v in pred_edges)
        orientation_accuracy = correct_orient / len(pred_edges) if pred_edges else 0.0

        metrics = {
            "SHD": shd,
            "F1": f1,
            "EdgeOrientationAccuracy": orientation_accuracy,
        }
        return {"total": sum(metrics.values()), "metrics": metrics}

    def evaluate_behavior(self):
        nE = sum(1 for _, _, action, _ in self.history if isinstance(action, dict))
        nT = sum(
            len(action) for _, _, action, _ in self.history if isinstance(action, dict)
        )
        nR = len(set(round_id for round_id, *_ in self.history))

        PE = self.e * nE
        PT = self.t * nT
        PR = self.r * nR

        total = self.alpha * PE + self.beta * PT + self.gamma * PR
        metrics = {"PE": PE, "PT": PT, "PR": PR}
        return {"total": total, "metrics": metrics}

    def evaluate_all(self):
        perf = self.PerformanceMetrics(self.history).evaluate()
        behav = self.BehaviorMetrics(self.history).evaluate()
        return {
            "performance": perf,
            "behavior": behav,
            "global_score": 0.5 * perf["total"] - 0.5 * behav["total"],
        }
