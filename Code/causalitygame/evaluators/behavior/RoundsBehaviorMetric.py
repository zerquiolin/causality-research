# Abstract
from ..abstract import BehaviorMetric

# Science
import pandas as pd

# Scripts
from causalitygame.lib.utils.metrics import log_penalty

# Constants
from causalitygame.lib.constants.environment import (
    ACTION_COLUMN,
    STOP_WITH_ANSWER_ACTION,
)


class RoundsBehaviorMetric(BehaviorMetric):
    """
    Compute a penalty based on the number of rounds taken before stopping with an answer.

    This metric counts all actions except the final 'stop_with_answer' and applies a logarithmic
    penalty function to the count.
    """

    name: str = "Rounds Behavior Metric"

    def __init__(self, alpha: float = 0.10) -> None:
        """
        Initialize the metric with a penalty decay parameter.

        Args:
            alpha (float): Decay rate for the log_penalty function; must be in (0, 1).

        Raises:
            ValueError: If alpha is not between 0 and 1 (exclusive).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

    def evaluate(self, history: pd.DataFrame) -> float:
        """
                Calculate the behavior metric for a given interaction history.
        This method counts the number of rounds in which the action was not 'stop_with_answer',
                excluding the final round, and applies the log_penalty function.

                Args:
                    history (pd.DataFrame): Interaction history, must contain an 'action' column.

                Returns:
                    float: The computed penalty metric.

                Raises:
                    TypeError: If history is not a pandas DataFrame.
                    KeyError: If the 'action' column is missing from the DataFrame.
        """
        # Validate input type
        if not isinstance(history, pd.DataFrame):
            raise TypeError(
                f"Expected history to be a pandas DataFrame, got {type(history)}"
            )

        # Ensure required column is present
        if ACTION_COLUMN not in history.columns:
            raise KeyError("The history DataFrame must contain an 'action' column.")

        # Exclude the final round (presumed to be the 'stop_with_answer' event)
        actions = history[ACTION_COLUMN].iloc[:-1]

        # Count rounds where the user did not stop with an answer
        non_stop_rounds = int((actions != STOP_WITH_ANSWER_ACTION).sum())

        # Apply logarithmic penalty
        penalty_value = log_penalty(non_stop_rounds, alpha=self.alpha)
        return penalty_value
