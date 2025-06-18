# Abstract
from ..abstract import DeliverableMetric

# Network
import networkx as nx

# Science
import pandas as pd

# Types
from typing import Hashable, Iterable, Set
from causalitygame.scm.abstract import SCM

# Constants
from causalitygame.lib.constants.environment import CURRENT_RESULT_COLUMN


def skeleton_edge_set(G: nx.DiGraph, nodes: Iterable[Hashable]) -> Set[frozenset]:
    """
    Compute the undirected edge set (“skeleton”) of a directed graph G
    restricted to a given node set.

    Args:
        G: A NetworkX directed graph.
        nodes: Iterable of node labels to include.

    Returns:
        A set of frozensets, each representing an undirected edge
        {u, v}.
    """
    node_set = set(nodes)
    # Only consider edges where both endpoints are in node_set
    edges = (frozenset((u, v)) for u, v in G.edges() if u in node_set and v in node_set)
    return set(edges)


class SHDDeliverableMetric(DeliverableMetric):
    """
    Deliverable metric that computes the Structural Hamming Distance (SHD)
    between the true SCM DAG and the final predicted graph result.
    """

    name: str = "Structural Hamming Distance Deliverable Metric"

    def evaluate(self, scm: SCM, history: pd.DataFrame) -> float:
        """
        Evaluate SHD between the SCM's true DAG and the predicted graph
        from the last history entry.

        Args:
            scm: An object containing the ground-truth DAG in `scm.dag.graph`.
            history: A pandas DataFrame with at least one row and a column
                     "current_result" holding NetworkX DiGraph predictions.

        Returns:
            The SHD as a non-negative integer.

        Raises:
            ValueError: If history is empty or lacks "current_result".
            TypeError:  If retrieved graphs are not nx.DiGraph instances.
        """
        if history.empty:
            raise ValueError("Cannot evaluate SHD: `history` is empty.")

        if CURRENT_RESULT_COLUMN not in history.columns:
            raise ValueError(
                f"Cannot evaluate SHD: '{CURRENT_RESULT_COLUMN}' column missing."
            )

        G_pred = history.iloc[-1][CURRENT_RESULT_COLUMN]
        G_true = scm.dag

        if not isinstance(G_true, nx.DiGraph) or not isinstance(G_pred, nx.DiGraph):
            raise TypeError(
                "Both true and predicted graphs must be NetworkX DiGraph instances."
            )

        # Union of nodes
        all_nodes = set(G_true.nodes()) | set(G_pred.nodes())

        true_skel = skeleton_edge_set(G_true, all_nodes)
        pred_skel = skeleton_edge_set(G_pred, all_nodes)

        # Symmetric difference = edges present in one but not both
        return len(true_skel.symmetric_difference(pred_skel))
