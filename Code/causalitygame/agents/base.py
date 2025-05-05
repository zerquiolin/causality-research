from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(
        self, samples: dict, actions: dict, num_rounds: int
    ) -> tuple[str, list]:
        """
        Choose an action based on the current state of the game.

        Parameters
        ----------
        samples : dict
            The samples collected so far.
        actions : dict
            The available actions to choose from.
        num_rounds : int
            The current round number.

        Returns
        -------
        tuple
            A tuple containing:
            - action: The chosen action.
            - treatments (list): A list of treatments to apply.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def submit_answer(self):
        """
        Returns the answer to the game.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
