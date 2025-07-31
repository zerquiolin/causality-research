# Abstract
from ..abstract import DeliverableMetric

# Types
import numbers
from typing import Any, Tuple, Union
from causalitygame.scm.abstract import SCM

Numeric = Union[int, float]


class AbsoluteErrorDeliverableMetric(DeliverableMetric):
    """
    A deliverable metric that calculates the absolute error between
    an actual value and a predicted value.
    """

    name: str = "Absolute Error Deliverable Metric"

    def mount(self, scm: SCM) -> None:  # unused
        pass

    def evaluate(self, scm: Any, data: Tuple[Numeric, Numeric]) -> float:
        """
        Evaluate the metric.

        Args:
            scm: Context or model object (unused in this metric, but required by interface).
            data: A tuple of two numbers (actual, predicted).

        Returns:
            The absolute error between actual and predicted values.

        Raises:
            TypeError: If data is not a tuple of two numeric types.
            ValueError: If the tuple does not contain exactly two elements.
        """
        # Validate that data is a tuple
        if not isinstance(data, tuple):
            raise TypeError(
                f"`data` must be a tuple of two numbers, got {type(data).__name__}"
            )

        # Validate tuple length
        if len(data) != 2:
            raise ValueError(f"`data` must have exactly two elements, got {len(data)}")

        actual, predicted = data

        # Validate element types
        for name, value in (("actual", actual), ("predicted", predicted)):
            if not isinstance(value, numbers.Number):
                raise TypeError(
                    f"`{name}` must be int or float, got {type(value).__name__}"
                )

        return abs(actual - predicted)
