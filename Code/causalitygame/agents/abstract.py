# Abstract
from abc import ABC, abstractmethod

# Types
from typing import Any, Dict, List, Tuple


class BaseAgent(ABC):
    """Abstract base class defining the interface for all game-playing agents."""

    @abstractmethod
    def inform(
        self, goal: Dict[str, Any], behavior_metric: str, deliverable_metric: str
    ) -> None:
        """Initialize the agent with the game goal and metrics.

        Args:
            goal: A dictionary specifying the target goal parameters.
            behavior_metric: The name of the metric to track agent behavior.
            deliverable_metric: The name of the metric to evaluate outcomes.
        """
        ...

    @abstractmethod
    def choose_action(
        self, samples: Dict[str, Any], actions: Dict[str, Any], num_rounds: int
    ) -> Tuple[str, List[Any]]:
        """Select the next action and associated treatments.

        Args:
            samples: Collected samples so far in the game.
            actions: Mapping of available actions the agent can take.
            num_rounds: Current round number (1-based).

        Returns:
            A tuple where:
            - first element is the chosen action key.
            - second element is a list of treatment identifiers to apply.
        """
        ...

    @abstractmethod
    def submit_answer(self) -> Any:
        """Generate the agent's final answer at game end.

        Returns:
            The answer in whatever form the game expects.
        """
        ...
