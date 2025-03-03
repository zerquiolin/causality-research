from abc import ABC, abstractmethod


class Player(ABC):
    """
    Abstract base class for a player in the game.
    Defines the structure and methods that any specific player should implement.
    """

    @abstractmethod
    def __init__(self, environment):
        """
        Initialize the player with a reference to the game.
        """
        self.environment = environment

    @abstractmethod
    def select_action(self):
        """
        Allow the player to select an action from the available actions.
        """
        pass

    # ? Maybe useful maybe not
    @abstractmethod
    def receive_feedback(self, next_state):
        """
        Update the player's strategy or knowledge based on feedback from the environment.
        Args:
            reward: The reward received after taking an action.
            next_state: The state of the environment after the action was applied.
        """
        pass

    @abstractmethod
    def adjust_strategy(self):
        """
        Adjust the player's strategy based on the outcome of previous actions.
        """
        pass

    @abstractmethod
    def infer_model(self, data):
        """
        Infer a causal Bayesian Network from the provided data using the causality library.
        Args:
            data: The dataset to be used for model inference.
        """
        pass
