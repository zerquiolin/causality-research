# Abstract
from ..abstract import DeliverableMetric

# Network
import networkx as nx

# Science
import numpy as np
import pandas as pd

# Types
from causalitygame.scm.abstract import SCM

# Constants
from causalitygame.lib.constants.environment import CURRENT_RESULT_COLUMN


class F1DeliverableMetric(DeliverableMetric):
    """
    Computes the F1 score between the true DAG and the predicted DAG stored in the SCM history.

    This metric considers all possible directed edges (excluding self-loops) over the union of nodes
    in both graphs, and calculates precision, recall, and their harmonic mean (F1 score).
    """

    name: str = "F1 Deliverable Metric"

    def evaluate(self, scm: SCM, history: pd.DataFrame) -> float:
        """
        Evaluate the F1 score for a predicted graph against the true graph in the SCM.

        Args:
            scm (Any): An object containing the ground-truth DAG under `scm.dag.graph` (a networkx.DiGraph).
            history (pd.DataFrame): DataFrame of evaluation history with a 'current_result' column holding
                the predicted nx.DiGraph in its last row.

        Returns:
            float: The F1 score (0.0 if no possible edges or in cases of zero precision/recall denominators).

        Raises:
            TypeError: If `history` is not a pandas DataFrame.
            ValueError: If `history` is empty.
            KeyError: If `history` lacks the 'current_result' column.
        """
        # Validate inputs
        if not isinstance(history, pd.DataFrame):
            raise TypeError(f"history must be a pandas DataFrame, got {type(history)}")
        if history.empty:
            raise ValueError("history DataFrame is empty; cannot evaluate metric.")
        if CURRENT_RESULT_COLUMN not in history.columns:
            raise KeyError(
                f"history must contain a '{CURRENT_RESULT_COLUMN}' column with predicted graphs."
            )

        # Extract true and predicted graphs
        true_graph: nx.DiGraph = scm.dag.graph
        pred_graph: nx.DiGraph = history.iloc[-1][CURRENT_RESULT_COLUMN]

        # Union of nodes
        nodes = sorted(set(true_graph.nodes()).union(pred_graph.nodes()))
        n = len(nodes)
        if n <= 1:
            return 0.0  # No possible edges

        # Create adjacency matrices over the same node ordering
        true_adj = nx.to_numpy_array(true_graph, nodelist=nodes, dtype=int)
        pred_adj = nx.to_numpy_array(pred_graph, nodelist=nodes, dtype=int)

        # Exclude self-loops by masking diagonal
        mask = ~np.eye(n, dtype=bool)
        true_flat = true_adj[mask]
        pred_flat = pred_adj[mask]

        # Compute counts
        tp = int(np.sum((true_flat == 1) & (pred_flat == 1)))
        fp = int(np.sum((true_flat == 0) & (pred_flat == 1)))
        fn = int(np.sum((true_flat == 1) & (pred_flat == 0)))

        # Avoid division by zero
        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
