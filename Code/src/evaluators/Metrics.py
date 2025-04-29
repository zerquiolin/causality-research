# Math
import numpy as np

# Graph
import networkx as nx

# Types
from typing import Dict


class BehaviorMetrics:
    def __init__(self, e=1.0, t=0.5, r=0.2, alpha=0.3, beta=0.5, gamma=0.2):
        self.e = e
        self.t = t
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.score = 0
        self.metrics = {
            "PE": 0.0,
            "PT": 0.0,
            "PR": 0.0,
        }

    def evaluate(self, history):
        nE = 0
        nT = 0
        nR = 0

        for i, row in history.iterrows():
            action = row["action"]
            action_object = row["action_object"]

            if action == "stop_with_answer":
                continue

            # Current experiment
            cne = len(action_object)
            # Current treatment
            cnt = sum(t[1] for t in action_object)

            # Update the number of experiments and treatments
            nE += cne
            nT += cnt
            nR += 1

        PE = self.e * nE
        PT = self.t * nT
        PR = self.r * nR

        self.score = self.alpha * PE + self.beta * PT + self.gamma * PR
        self.metrics.update({"PE": PE, "PT": PT, "PR": PR})

        return {"score": self.score, "metrics": self.metrics}


class DeliverableMetrics:
    def __init__(self, scm, w1=0.5, w2=0.2, w3=0.3):
        self.scm = scm
        assert w1 + w2 + w3 == 1.0, "Weights must sum to 1"
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.score = 0
        self.metrics = {
            "SHD": 0.0,
            "F1": 0.0,
            "EdgeOrientationAccuracy": 0.0,
        }

    def evaluate(self, history, goal_type):
        if goal_type == "dag_discovery":
            return self._evaluate_dag_discovery(history)

        # Error handling for unsupported goal types
        raise NotImplementedError(f"Goal '{goal_type}' not supported.")

    def _evaluate_dag_discovery(self, history: Dict):
        # Get both graphs
        G_true: nx.DiGraph = self.scm.dag.graph
        G_pred: nx.DiGraph = history.iloc[-1]["action_object"]
        # Check if the true and predicted graph is valid
        if not isinstance(G_true, nx.DiGraph) or not isinstance(G_pred, nx.DiGraph):
            self.metrics.update(
                {
                    "SHD": 0.0,
                    "F1": 0.0,
                    "EdgeOrientationAccuracy": 0.0,
                }
            )
            return {"score": 0, "metrics": self.metrics}

        # Ensure same nodes
        nodes = sorted(set(G_true.nodes()).union(set(G_pred.nodes())))
        G_true = nx.relabel_nodes(G_true.subgraph(nodes), lambda x: x)
        G_pred = nx.relabel_nodes(G_pred.subgraph(nodes), lambda x: x)

        # Edge sets
        true_edges = set(G_true.edges())
        pred_edges = set(G_pred.edges())

        # 1. Structural Hamming Distance (SHD)
        undirected_true = set(map(frozenset, true_edges))
        undirected_pred = set(map(frozenset, pred_edges))
        undirected_diff = undirected_true.symmetric_difference(undirected_pred)
        shd = len(undirected_diff)

        # 2. F1 Score
        y_true = [
            1 if (u, v) in true_edges else 0 for u in nodes for v in nodes if u != v
        ]
        y_pred = [
            1 if (u, v) in pred_edges else 0 for u in nodes for v in nodes if u != v
        ]

        f1 = self._compute_f1_score(y_true, y_pred)

        # 3. Edge Orientation Accuracy
        correct_orientations = sum((u, v) in true_edges for (u, v) in pred_edges)
        orientation_accuracy = (
            correct_orientations / len(pred_edges) if pred_edges else 0.0
        )

        # Total score
        total = self.w1 * shd + self.w2 * f1 + self.w3 * orientation_accuracy
        # Metrics
        metrics = {
            "SHD": shd,
            "F1": f1,
            "EdgeOrientationAccuracy": orientation_accuracy,
        }

        return {
            "score": total,
            "metrics": metrics,
        }

    def _compute_f1_score(self, y_true, y_pred, zero_division=0):
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


class Metrics:
    def __init__(
        self,
        behavior_metric: BehaviorMetrics,
        deliverable_metric: DeliverableMetrics,
        goal_type: str,
        plambda: float = 0.8,
    ):
        self.behavior_metric = behavior_metric
        self.deliverable_metric = deliverable_metric
        self.goal_type = goal_type
        self.plambda = plambda

    def evaluate(self, history):
        # Evaluate behavior metrics
        behavior = self.behavior_metric.evaluate(history)
        # Evaluate deliverable metrics
        deliverable = self.deliverable_metric.evaluate(
            history=history, goal_type=self.goal_type
        )

        # Combine metrics
        combined_metrics = {
            "behavior": behavior,
            "deliverable": deliverable,
            "global_score": self.plambda * deliverable["score"]
            + (1 - self.plambda) * behavior["score"],
        }
        # Return combined metrics
        return combined_metrics
