# Abstract
from ..abstract import DeliverableMetric

# Science
import numpy as np

# Types
from typing import Sequence, Optional


class MeanSquaredErrorDeliverableMetric(DeliverableMetric):
    """
    Computes the Mean Squared Error (MSE) between two sequences of numeric values.

    MSE = mean((prediction_i - target_i)^2) for all i.

    Attributes:
        name (str): Human-readable name for the metric.
    """

    name: str = "Mean Squared Error Deliverable Metric"

    def evaluate(self, scm: Optional[object], data: Sequence[Sequence[float]]) -> float:
        """
        Evaluate the metric.

        Args:
            scm: A scenario/context object (unused for MSE).
            data: A two-element sequence: (predictions, targets). Each must be a sequence of floats,
                  and both sequences must have the same length.

        Returns:
            float: The computed mean squared error.

        Raises:
            TypeError: If `data` is not a sequence of two sequences of floats.
            ValueError: If the two sequences differ in length.
        """
        # Unpack predictions and targets
        try:
            preds, targets = data
        except (ValueError, TypeError):
            raise TypeError(
                "`data` must be a sequence of two sequences: (predictions, targets)."
            )

        # Convert inputs to NumPy arrays
        preds_arr = np.asarray(preds, dtype=float)
        targets_arr = np.asarray(targets, dtype=float)

        # Validate shapes
        if preds_arr.shape != targets_arr.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape; "
                f"got {preds_arr.shape} vs {targets_arr.shape}."
            )

        # Core computation: vectorized MSE
        squared_errors = (preds_arr - targets_arr) ** 2
        mse_value = float(np.mean(squared_errors))

        return mse_value
