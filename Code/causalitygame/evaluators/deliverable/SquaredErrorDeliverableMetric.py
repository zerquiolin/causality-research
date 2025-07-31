# Abstract
from ..abstract import DeliverableMetric

# Types
from typing import Any, Tuple, Union
from causalitygame.scm.abstract import SCM

Numeric = Union[int, float]


class SquaredErrorDeliverableMetric(DeliverableMetric):
    """
    DeliverableMetric implementation computing squared error between two values.

    Attributes:
        name (str): Human-readable name of the metric.
    """

    name: str = "Squared Error Deliverable Metric"

    def mount(self, scm: SCM) -> None:  # unused
        pass

    def evaluate(self, scm: Any, data: Tuple[Numeric, Numeric]) -> float:
        """
        Evaluate the squared error for a given pair of values.

        This implementation unpacks a tuple of (actual, predicted) values
        and delegates the computation to the `squared_error` helper.

        Args:
            scm (Any): Scenario or context object (unused in this metric).
            data (Tuple[Numeric, Numeric]): Pair of values (actual, predicted).

        Returns:
            float: The squared error between actual and predicted.

        Raises:
            TypeError: If data is not a tuple of length 2 or contains non-numeric types.
            ValueError: If data tuple does not contain exactly two elements.
        """
        if not isinstance(data, tuple):
            raise TypeError(f"Expected data as tuple, got {type(data).__name__}")
        if len(data) != 2:
            raise ValueError(
                f"Data tuple must have exactly two elements, got {len(data)}"
            )

        actual, predicted = data
        try:
            return (float(actual) - float(predicted)) ** 2
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Both actual and predicted values must be numeric types: {e}"
            ) from e
