import numpy as np
import pandas as pd

from ..base import BaseAgent

# Dag Learning Script
from causalitygame.lib.scripts.pc import learn as learn_dag
from causalitygame.lib.scripts.empiricalCATE import compute_empirical_cate_fuzzy
from causalitygame.lib.scripts.xgboostTE import te_estimation

from typing import Dict


class ExhaustiveAgent(BaseAgent):
    _is_first_round = True
    _is_numeric = False

    def inform(self, goal: str, behavior_metric: str, deliverable_metric: str):
        """
        Inform the agent about the goal, behavior metric, and deliverable.
        """
        # Store the goal, behavior metric, and deliverable
        self._process = goal.split(":")[0]
        self._goal = goal
        self._behavior_metric = behavior_metric
        self._deliverable_metric = deliverable_metric
        self.past_data: pd.DataFrame = None

    def choose_action(self, samples, actions, num_rounds):
        # Save the samples for future analysis
        self._merge_data(samples)

        # Define the treatment list
        treatments = []
        num_obs = 1  # 10**4
        num_inter = 1  # 10**4
        # Iterate over all possible actions
        for node in actions.keys():
            # Skip the stop_with_answer action
            if node == "stop_with_answer" or node == "Y" or node == "X":
                continue

            # Generate observation data
            if node == "observe":
                treatments.append(("observe", num_obs))
                continue

            # Get the domain of the action
            domain = actions[node]
            # Check if the domain is categorical
            if type(domain[0]) is str:
                # if isinstance(domain, list):
                for value in domain:
                    # Add all possible values to the treatment list
                    treatments.append(({node: value}, num_inter))
            else:
                self._is_numeric = True
                if isinstance(domain, list):
                    # If the domain is a list, we assume it is a discrete set of values
                    treatments.append(({node: domain[0]}, num_inter))
                    treatments.append(({node: domain[1]}, num_inter))
                else:
                    # If the domain is a tuple, we assume it is a range
                    for i in np.linspace(domain[0], domain[1], 10):
                        # Add 10 values to the treatment list
                        treatments.append(({node: i}, num_inter))

        return "experiment", treatments

    def _merge_data(self, new_data: Dict[tuple, pd.DataFrame]):
        # Check if new_data is empty
        if not new_data:
            self.past_data = pd.DataFrame()
            return

        # Reset the past data if this is the first round
        self.past_data = pd.DataFrame(columns=list(new_data.values())[0].columns)

        for treatment, df in new_data.items():
            # Add the new data to the historical data
            self.past_data = pd.concat([self.past_data, df], ignore_index=True)

    def submit_answer(self):
        task_mapping = {
            "DAG Inference Mission": self._dag_inference_task,
            "Conditional Average Treatment Effect (CATE) Mission": self._cate_task,
            "Treatment Effect Mission": self._te_task,
        }
        return task_mapping.get(self._process, None)()

    def _dag_inference_task(self):
        return learn_dag(self.past_data.copy(), self._is_numeric)

    def _cate_task(self):
        data = (self.past_data.copy(),)

        def compute_cate(Y, T, Z):
            return compute_empirical_cate_fuzzy(
                query={"Y": Y, "T": T, "Z": Z},
                data=data,
                distance_threshold=10**2,
            )

        return compute_cate

    def _te_task(self):
        data = self.past_data.copy()

        def compute_te(Y, Z, X):
            """
            Compute the treatment effect given Y, Z, and X.
            """
            data.to_csv("past_data.csv", index=False)  # Save past data for debugging
            return te_estimation(
                Y=Y,
                Z=Z,
                X=X,
                data=data,
            )

        return compute_te
