# Abstract
from ..abstract import DeliverableMetric

# Network
import networkx as nx

# Science
import pandas as pd

# Types
from causalitygame.scm.abstract import SCM

# Constants
from causalitygame.lib.constants.environment import CURRENT_RESULT_COLUMN


class EdgeAccuracyDeliverableMetric(DeliverableMetric):
    """
    Deliverable metric that measures the precision of predicted edges
    in a learned DAG relative to a ground-truth SCM DAG.
    """

    name: str = "Edge Precision Metric"

    def evaluate(self, scm: SCM, history: pd.DataFrame) -> float:
        """
        Evaluate and return the precision of the final predicted graph.

        Args:
            scm: A structural causal model object with attribute `dag.graph`
                 as a networkx.DiGraph of the ground-truth.
            history: A pandas DataFrame of past interactions, which must
                     include a 'current_result' column holding a DiGraph
                     in its last row.

        Returns:
            float: Fraction of predicted edges that are present in the
                   true graph (0.0 if no predicted edges).

        Raises:
            ValueError: If history is empty.
            KeyError: If 'current_result' is missing from history.
            TypeError: If the true or predicted graph is not a DiGraph.
        """
        # Validate history
        if history.empty:
            raise ValueError("Cannot evaluate metric on empty history.")
        if CURRENT_RESULT_COLUMN not in history.columns:
            raise KeyError("Expected 'current_result' column in history DataFrame.")

        # Extract graphs
        G_pred = history.iloc[-1][CURRENT_RESULT_COLUMN]
        if not isinstance(G_pred, nx.DiGraph):
            raise TypeError(
                f"Predicted result must be a networkx.DiGraph, got {type(G_pred)}."
            )

        G_true = scm.dag.graph
        if not isinstance(G_true, nx.DiGraph):
            raise TypeError(
                f"True graph must be a networkx.DiGraph, got {type(G_true)}."
            )

        # Compute precision
        true_edges = set(G_true.edges())
        pred_edges = set(G_pred.edges())
        if not pred_edges:
            return 0.0
        correct = len(true_edges & pred_edges)
        return correct / len(pred_edges)
