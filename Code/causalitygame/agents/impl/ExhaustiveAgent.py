import numpy as np

from ..base import BaseAgent

# Dag Learning Script
from causalitygame.lib.scripts.pc import learn as learn_dag

from typing import Any, Dict, List, Tuple


class ExhaustiveAgent(BaseAgent):
    _is_first_round = True
    _is_numeric = False

    def inform(self, goal: str, behavior_metric: str, deliverable_metric: str):
        """
        Inform the agent about the goal, behavior metric, and deliverable.
        """
        # Store the goal, behavior metric, and deliverable
        self._goal = goal
        self._behavior_metric = behavior_metric
        self._deliverable_metric = deliverable_metric
        self.past_data: Dict[str, Any] = {
            "empty": [],
        }

    def choose_action(self, samples, actions, num_rounds):
        # Save the samples for future analysis
        self._merge_data(samples)
        # Check if this is the first round
        if self._is_first_round:
            self._is_first_round = False
        else:
            return "stop_with_answer", None

        # Define the treatment list
        treatments = []
        num_obs = 10**4
        num_inter = 10**4
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
                self._is_numeric = True
                for i in np.linspace(domain[0], domain[1], 10):
                    # Add 10 values to the treatment list
                    treatments.append(({node: i}, num_inter))

        return "experiment", treatments

    def _merge_data(self, new_data: Dict[str, Dict[str, List[Any]]]):
        for key, value in new_data.items():
            if key == "empty":
                self.past_data["empty"].extend(value)
            else:
                if key not in self.past_data:
                    self.past_data[key] = {}
                for val, records in value.items():
                    if val not in self.past_data[key]:
                        self.past_data[key][val] = []
                    self.past_data[key][val].extend(records)

    def submit_answer(self):
        return learn_dag(self.past_data, self._is_numeric)
