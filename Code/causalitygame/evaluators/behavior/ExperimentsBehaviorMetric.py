# Abstract
from ..abstract import BehaviorMetric

# Science
import pandas as pd

# Scripts
from causalitygame.lib.utils.metrics import log_penalty

# Constants
from causalitygame.lib.constants.environment import (
    ACTION_COLUMN,
    ACTION_OBJECT_COLUMN,
    STOP_WITH_ANSWER_ACTION,
)


class ExperimentsBehaviorMetric(BehaviorMetric):
    """
    Behavior metric based on counting experimental actions in the history log.
    Applies a logarithmic penalty to encourage concise experimentation.
    """

    name: str = "Experiments Behavior Metric"

    def __init__(self, alpha: float = 0.30) -> None:
        """
        Initialize the experiment behavior metric.

        Args:
            alpha: Penalty factor passed to the log_penalty function.
        """
        self.alpha = alpha

    def evaluate(self, history: pd.DataFrame) -> float:
        """
        Compute the experiments-based behavior metric.

        Args:
            history: A pandas DataFrame with at least the columns:
                - 'action': str identifiers of actions taken.
                - 'action_object': iterable containers per action.

        Returns:
            A non-negative float penalty score, computed via log_penalty.

        Raises:
            ValueError: If the history DataFrame is malformed.
        """
        # Return zero penalty if there's no history
        if history is None or history.empty:
            return 0.0

        # Exclude the final action from counting
        filtered = history.iloc[:-1]

        # Mask out any terminal actions
        mask = filtered[ACTION_COLUMN] != STOP_WITH_ANSWER_ACTION

        try:
            # Sum the lengths of the action_object iterables
            nE = (
                filtered.loc[mask, ACTION_OBJECT_COLUMN]
                .apply(lambda obj: len(obj) if hasattr(obj, "__len__") else 0)
                .sum()
            )
        except Exception as exc:
            raise ValueError(f"Invalid history format: {exc}")

        # Apply logarithmic penalty
        return log_penalty(nE, alpha=self.alpha)
