from abc import ABC, abstractmethod


class MDP(ABC):
    """
    Abstract base class for an MDP (Markov Decision Process).
    Defines the structure and methods that any specific MDP should implement.
    """

    @abstractmethod
    def __init__(self, states, actions):
        """
        Initialize the MDP with a set of states and actions.

        Parameters:
        - states (list): A list of possible states in the MDP.
        - actions (list): A list of possible actions in the MDP.
        """
        self.states = states
        self.actions = actions

    @abstractmethod
    def is_terminal(self):
        """
        Check if the game has reached its terminal state.
        Returns True if there are no more interventions left, False otherwise.
        """
        pass

    @abstractmethod
    def step(self, action, value, return_object=False):
        """
        Apply an intervention (action) to the environment, updating the state.

        Args:
            action: The intervention to be applied.
            value: The value associated with the intervention.
            return_object (optional): Whether to return the updated state object. Defaults to False.

        Raises:
            ValueError: If the action is not a valid intervention.

        Returns:
            None or State: If return_object is True, returns the updated state object. Otherwise, returns None.
        """
        pass

    @abstractmethod
    def available_actions(self):
        """
        Return the list of available actions (interventions) left in the environment.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        pass
