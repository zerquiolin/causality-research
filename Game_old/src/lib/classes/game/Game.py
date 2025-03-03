from src.lib.classes.abstract.MDP import MDP
from src.lib.classes.abstract.Player import Player


class CausalityGame:
    """
    This class manages the gameplay by combining the environment and player.
    The player interacts with the environment to play the game.
    """

    def __init__(self, environment: MDP, player: Player):
        self.environment = environment
        self.player = player

    def play(self):
        """
        Play the game by letting the player select actions and apply them to the environment.
        """
        while not self.environment.is_terminal():
            intervention, value = self.player.select_action()
            if intervention is not None or value is None:
                self.environment.step(intervention, value)
        print("Game over! All interventions have been performed.")

    def print_state(self):
        """
        Get the current state of the environment.
        """

        print("Current state of the environment:")
        print(self.environment.state)

        for key, value in self.environment.state.items():
            for key2, value2 in value.items():
                print(f"{key} = {key2}")
                print(value2)
