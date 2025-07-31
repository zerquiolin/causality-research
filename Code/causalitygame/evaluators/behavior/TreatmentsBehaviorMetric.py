# Abstract
from ..abstract import BehaviorMetric

# Science
import pandas as pd

# Scripts
from causalitygame.lib.utils.metrics import log_penalty

# Types
from causalitygame.scm.abstract import SCM

# Constants
from causalitygame.lib.constants.environment import (
    ACTION_COLUMN,
    ACTION_OBJECT_COLUMN,
    STOP_WITH_ANSWER_ACTION,
)


class TreatmentsBehaviorMetric(BehaviorMetric):
    """
    Calculates a behavior metric based on the number of treatments applied
    before the final answer. Applies a logarithmic penalty to the total.
    """

    name = "Treatments Behavior Metric"

    def __init__(self, alpha: float = 0.20):
        """
        Initialize the metric.

        Args:
            alpha (float): Penalty scaling factor for log_penalty.
        """
        self.alpha = alpha

    def mount(self, scm: SCM) -> None:  # unused
        pass

    def evaluate(self, history: pd.DataFrame) -> float:
        """
        Evaluate the behavior metric for a given interaction history.

        This method counts the number of treatments (sum of the second element
        in each action_object tuple) for all rounds except the final one,
        excluding any round where the action is 'stop_with_answer',
        and applies the log_penalty function.

        Args:
            history (pd.DataFrame): Interaction history with columns:
                - 'action': str identifier of the action taken.
                - 'action_object': Iterable of tuples where the second element
                                   is a numeric treatment count.

        Returns:
            float: The computed penalty metric.

        Raises:
            TypeError: If history is not a pandas DataFrame.
            ValueError: If 'action_object' entries are not iterable of tuples.
        """
        if not isinstance(history, pd.DataFrame):
            raise TypeError(
                f"Expected history as pandas.DataFrame, got {type(history)}"
            )

        # Filter out the final round and any 'stop_with_answer' actions
        filtered = history.iloc[:-1]
        filtered = filtered[filtered[ACTION_COLUMN] != STOP_WITH_ANSWER_ACTION]

        # Sum the treatment counts
        n_treatments = 0
        for idx, row in filtered.iterrows():
            action_obj = row[ACTION_OBJECT_COLUMN]
            try:
                # Expect iterable of tuples, second element numeric
                counts = (item[1] for item in action_obj)
                n_treatments += sum(counts)
            except Exception as e:
                raise ValueError(f"Invalid action_object at index {idx}: {e}")

        # Apply logarithmic penalty
        return log_penalty(n_treatments, alpha=self.alpha)
