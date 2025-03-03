# Libs
import random

# Modules
from src.lib.classes.abstract.Player import Player

# Causality
from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest, ChiSquaredTest

# Causal-learn
from causallearn.search.ConstraintBased.PC import pc


class RandomPlayer(Player):
    """
    A player that selects random actions from the list of available interventions.
    Also capable of inferring causal models from data.
    """

    def __init__(self, environment):
        super().__init__(environment)

    def select_action(self):
        """
        Randomly selects an available intervention from the game.
        """
        if self.environment.is_terminal():
            print("All interventions have been performed.")
            return None

        intervention = random.choice(list(self.environment.available_actions().keys()))
        value = random.choice(self.environment.available_actions()[intervention])
        return (intervention, value)

    def receive_feedback(self):
        pass

    def adjust_strategy(self):
        """
        A random player does not adjust strategy, but this method is here for future extensions.
        """
        print("Random player does not adjust strategy.")

    def infer_model(self, data, labels):
        """
        Infers a causal Bayesian Network from the provided dataset using the causality library.
        Args:
            data: A pandas DataFrame with the dataset for inference.
        Returns:
            model: The inferred causal Bayesian Network.
        """
        # run the search
        # ic_algorithm = IC(RobustRegressionTest)
        ic_algorithm = IC(ChiSquaredTest)
        graph = ic_algorithm.search(
            data=data,
            variable_types={node: "d" for node in labels},
        )
        print(graph.edges(data=True))

    def causal_learning(self, data, labels=None):
        """
        Perform causal learning using the PC algorithm.
        """

        # Perform causal discovery using the PC algorithm
        pc_result = pc(data, indep_test="chisq")

        print(pc_result.G)

        pc_result.draw_pydot_graph(labels=labels)
