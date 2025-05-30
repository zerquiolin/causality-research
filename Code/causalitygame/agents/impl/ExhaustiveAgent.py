import numpy as np
import pandas as pd

from ..base import BaseAgent

# Dag Learning Script
from causalitygame.lib.scripts.pc import learn as learn_dag
from causalitygame.lib.scripts.empiricalCATE import compute_empirical_cate_fuzzy
from causalitygame.lib.scripts.xgboostTE import te_estimation

from typing import Any, Dict, List, Tuple


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
        self.visited_nodes: set[tuple] = set()
        self.past_data: pd.DataFrame = None

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
            if type(domain[0]) is str:
                # if isinstance(domain, list):
                for value in domain:
                    # Add all possible values to the treatment list
                    treatments.append(({node: value}, num_inter))
            else:
                self._is_numeric = True
                for i in np.linspace(domain[0], domain[1], 10):
                    # Add 10 values to the treatment list
                    treatments.append(({node: i}, num_inter))

        return "experiment", treatments

    def _merge_data(self, new_data: Dict[tuple, pd.DataFrame]):
        # Check if new_data is empty
        if not new_data:
            return
        # Check if past_data is None
        if self.past_data is None:
            # Initialize past_data with the first data frame
            self.past_data = pd.DataFrame(columns=list(new_data.values())[0].columns)

        for treatment, df in new_data.items():
            # Check if the treatment is already seen
            if treatment in self.visited_nodes:
                continue
            # Add the new data to the historical data
            self.past_data = pd.concat([self.past_data, df], ignore_index=True)

            # Add the treatment to the visited nodes
            self.visited_nodes.add(treatment)

    def submit_answer(self):
        task_mapping = {
            "DAG Inference Mission": self._dag_inference_task,
            "Conditional Average Treatment Effect (CATE) Mission": self._cate_task,
            "Treatment Effect Mission": self._te_task,
        }
        return task_mapping.get(self._process, None)()

    def _dag_inference_task(self):
        return learn_dag(self.past_data, self._is_numeric)

    def _cate_task(self):
        def compute_cate(Y, T, Z):
            return compute_empirical_cate_fuzzy(
                query={"Y": Y, "T": T, "Z": Z},
                data=self.past_data,
                distance_threshold=10**2,
            )

        return compute_cate

    def _te_task(self):
        def compute_te(Y, Z, X):
            """
            Compute the treatment effect given Y, Z, and X.
            """
            return te_estimation(
                Y=Y,
                Z=Z,
                X=X,
                data=self.past_data,
            )

        return compute_te
