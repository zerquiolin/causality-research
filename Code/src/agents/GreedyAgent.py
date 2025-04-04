import re
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.estimators import PC
from pgmpy.models import BayesianModel

from .base import BaseAgent

# Dag Learning Script
from src.lib.scripts.dag import learn_dag


class GreedyAgent(BaseAgent):
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
        # Iterate over all possible actions
        for node in actions.keys():
            # Skip the stop_with_answer action
            if node == "stop_with_answer":
                continue

            # Generate observation data
            if node == "observe":
                treatments.append(("observe", 1500))
                continue

            # Get the domain of the action
            domain = actions[node]
            # Check if the domain is categorical
            if isinstance(domain, list):
                for value in domain:
                    # Add all possible values to the treatment list
                    treatments.append(({node: value}, 500))
            else:
                for i in np.linspace(domain[0], domain[1], 10):
                    # Add 10 values to the treatment list
                    treatments.append(({node: i}, 500))

        return "experiment", treatments

    def _analyze_dataset(self, samples):
        """
        Analyze the dataset and update the learned DAG.

        This method separates the DAG discovery into two phases:
        1. Build an initial DAG using only observational data.
        2. Refine edge orientations using interventional datasets.
        """
        # === Phase 1: Observational DAG ===
        df_obs = pd.DataFrame(data=samples["empty"])

        # === Phase 2: Orientation Refinement Using Interventional Data ===
        intervention_sets = {}
        for key, interventions in samples.items():
            if key == "empty":
                continue
            intervention_sets[key] = []
            for intervention_samples in interventions.values():
                intervention_sets[key].extend(intervention_samples)

        for key, intervention_samples in intervention_sets.items():
            intervention_sets[key] = pd.DataFrame(data=intervention_samples)

        learned_dag = learn_dag(
            df_obs=df_obs, interventions=intervention_sets, alpha=0.1
        )

        # Save the refined graph as the learned DAG.
        self._learned_graph = learned_dag

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """

        return self._learned_graph
