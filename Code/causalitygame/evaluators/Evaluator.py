# Abstract
from .abstract import DeliverableMetric, BehaviorMetric

# Science
import pandas as pd

# Types
from typing import List, Tuple, Union
from causalitygame.scm.abstract import SCM
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationResult:
    """
    Container for storing evaluation results.

    Attributes:
        behavior (List[Tuple[str, float]]): List of (metric_name, score) for behavior metrics.
        deliverable (List[Tuple[str, float]]): List of (metric_name, score) for deliverable metrics.
        global_score (float): Combined weighted score.
    """

    behavior: List[Tuple[str, float]] = field(default_factory=list)
    deliverable: List[Tuple[str, float]] = field(default_factory=list)
    global_score: float = 0.0


class Evaluator:
    """
    Evaluator computes and aggregates behavior and deliverable metrics
    based on provided SCM and interaction history.

    Each metric must implement an `evaluate` method. The final score
    is a weighted sum controlled by `plambda`.
    """

    def __init__(
        self,
        scm: SCM,
        behavior_metrics: List[BehaviorMetric] = [],
        deliverable_metrics: List[DeliverableMetric] = [],
        plambda: float = 0.8,
    ) -> None:
        # Validate inputs
        if not 0.0 <= plambda <= 1.0:
            raise ValueError(f"plambda must be between 0 and 1, got {plambda}")

        self.scm: SCM = scm
        self.behavior_metrics: List[BehaviorMetric] = behavior_metrics
        self.deliverable_metrics: List[DeliverableMetric] = deliverable_metrics
        self.plambda: float = plambda

    def evaluate(
        self,
        history: Union[pd.DataFrame, List[dict]],
    ) -> EvaluationResult:
        """
        Compute all behavior and deliverable metrics and aggregate into a global score.

        Args:
            history (Union[pd.DataFrame, List[dict]]): Interaction history data.

        Returns:
            EvaluationResult: Detailed scores and combined global score.

        Raises:
            TypeError: If history is not a DataFrame or list of dicts.
        """
        # Normalize history
        if isinstance(history, list):
            try:
                history_df = pd.DataFrame(history)
            except Exception as e:
                raise TypeError("Failed to convert history list to DataFrame") from e
        elif isinstance(history, pd.DataFrame):
            history_df = history
        else:
            raise TypeError(
                "History must be a pandas DataFrame or list of dictionaries"
            )

        # Evaluate each behavior metric
        behavior_scores: List[Tuple[str, float]] = []
        for metric in self.behavior_metrics:
            score = metric.evaluate(history_df)
            behavior_scores.append((metric.name, score))

        # Evaluate each deliverable metric
        deliverable_scores: List[Tuple[str, float]] = []
        for metric in self.deliverable_metrics:
            score = metric.evaluate(self.scm, history_df)
            deliverable_scores.append((metric.name, score))

        # Sum and weight
        total_behavior = sum(score for _, score in behavior_scores)
        total_deliverable = sum(score for _, score in deliverable_scores)
        weighted_behavior = total_behavior * (1 - self.plambda)
        weighted_deliverable = total_deliverable * self.plambda
        global_score = weighted_behavior + weighted_deliverable

        return EvaluationResult(
            behavior=behavior_scores,
            deliverable=deliverable_scores,
            global_score=global_score,
        )
