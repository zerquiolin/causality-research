
import random
from ..abstract.Player import Player
from causality.inference import search as causality_search
import pandas as pd

class GreedyPlayer(Player):
    """
    A player that selects actions to maximize immediate rewards.
    """
    def __init__(self, game):
        super().__init__(game)

    def select_action(self):
        """
        Select the action that currently offers the highest reward.
        """
        available_actions = self.game.available_actions()
        best_action = max(available_actions, key=lambda action: self.game.get_reward(self.game.state, action))
        return best_action

class BayesianPlayer(Player):
    """
    A player that uses Bayesian inference to guide its decisions.
    """
    def __init__(self, game):
        super().__init__(game)

    def select_action(self):
        """
        Use Bayesian inference to select the next action that maximizes information gain.
        """
        available_actions = self.game.available_actions()
        # Use Bayesian logic to select the best action (dummy implementation for now)
        return random.choice(available_actions)

    def infer_model(self, data):
        """
        Infers a causal Bayesian Network from the provided dataset using the causality library.
        """
        variables = data.columns
        print(f"Inferring causal model from data with variables: {variables}")
        results = causality_search.greedy_search(data)
        model = results['graph']
        print(f"Inferred causal model: {model}")
        return model
