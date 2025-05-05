import numpy as np

from ..base import BaseAgent

# Dag Learning Script
from causalitygame.lib.scripts.pc import learn as learn_dag


class ExhaustiveAgent(BaseAgent):
    _is_first_round = True

    def choose_action(self, samples, actions, num_rounds):
        # Check if this is the first round
        if self._is_first_round:
            self._is_first_round = False
        else:
            self._analyze_dataset(samples)
            return "stop_with_answer", None

        # Define the treatment list
        treatments = []
        num_obs = 10**3
        num_inter = 10**3
        # Iterate over all possible actions
        for node in actions.keys():
            # Skip the stop_with_answer action
            if node == "stop_with_answer":
                continue

            # Generate observation data
            if node == "observe":
                treatments.append(("observe", num_obs))
                continue

            # Get the domain of the action
            domain = actions[node]
            # Check if the domain is categorical
            if isinstance(domain, list):
                for value in domain:
                    # Add all possible values to the treatment list
                    treatments.append(({node: value}, num_inter))
            else:
                for i in np.linspace(domain[0], domain[1], 10):
                    # Add 10 values to the treatment list
                    treatments.append(({node: i}, num_inter))

        return "experiment", treatments

    def _analyze_dataset(self, samples):
        """
        Analyze the dataset and update the learned DAG.

        This method separates the DAG discovery into two phases:
        1. Build an initial DAG using only observational data.
        2. Refine edge orientations using interventional datasets.
        """
        # Use custom DAG learning script
        learned_dag = learn_dag(samples)
        # Save the refined graph as the learned DAG.
        self._learned_graph = learned_dag

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """

        return self._learned_graph
