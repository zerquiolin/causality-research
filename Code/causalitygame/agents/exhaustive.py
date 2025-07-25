# Abstract
from .abstract import BaseAgent

# Science
import numpy as np
import pandas as pd

# Constants
from causalitygame.lib.constants.environment import (
    OBSERVABLE_COLUMN,
    OBSERVE_ACTION,
    EXPERIMENT_ACTION,
    STOP_WITH_ANSWER_ACTION,
)

# Helpers
from causalitygame.lib.helpers.tasks import TaskFactory

# Types
from typing import Dict, Callable


class ExhaustiveAgent(BaseAgent):
    """
    An agent that enumerates all possible interventions exhaustively
    and then computes the requested causal or treatment‐effect task.
    """

    def __init__(self, num_obs: int = 1, num_inter: int = 1):
        super().__init__()
        self._goal = None
        self._behavior_metric = None
        self._deliverable_metric = None
        self.data: pd.DataFrame = pd.DataFrame()
        self._is_numeric = False
        self._num_obs = num_obs
        self._num_inter = num_inter

    def inform(self, goal, behavior_metric, deliverable_metric):
        self._goal = goal["goal"]
        self._behavior_metric = behavior_metric
        self._deliverable_metric = deliverable_metric
        self.data: pd.DataFrame = None

    def choose_action(self, samples, actions, num_rounds):
        # Save the samples for future analysis
        self._update_data(samples)

        # Define the treatment list
        treatments = []
        # Iterate over all possible actions
        for node in actions.keys():
            # Skip the stop_with_answer action
            if node == STOP_WITH_ANSWER_ACTION:
                continue

            # Generate observation data
            if node == OBSERVE_ACTION:
                treatments.append((OBSERVE_ACTION, self._num_obs))
                continue

            # Get the domain of the action
            domain = actions[node]
            # Check domain type
            if type(domain[0]) is str:  # Categorical domain
                for value in domain:
                    treatments.append(({node: value}, self._num_inter))
            else:  # Numerical domain
                self._is_numeric = True
                if isinstance(domain, list):  # Discrete numerical domain
                    treatments.append(({node: domain[0]}, self._num_inter))
                    treatments.append(({node: domain[1]}, self._num_inter))
                else:  # Continuous numerical domain
                    for i in np.linspace(domain[0], domain[1], 10):
                        # Add 10 values to the treatment list
                        treatments.append(({node: i}, self._num_inter))

        return EXPERIMENT_ACTION, treatments

    def _update_data(self, new_data: Dict[tuple, pd.DataFrame]):
        # Check if new_data is empty
        if not new_data:
            self.data = pd.DataFrame()
            return

        # Reset the past data (to avoid conflicts with previous rounds)
        self.data = pd.DataFrame(columns=list(new_data.values())[0].columns)

        # Filter out empty or all-NA DataFrames before concatenation
        valid_dfs = [
            df
            for treatment, df in new_data.items()
            if not df.empty
            and not df.isna().all().all()
            and treatment
            != OBSERVABLE_COLUMN  # TODO: Check if this is worth keeping (exists for te)
        ]
        if valid_dfs:
            # Add the new data
            self.data = pd.concat(valid_dfs, ignore_index=True)
        else:
            # Reset to an empty DataFrame with the correct columns
            self.data = pd.DataFrame(columns=list(new_data.values())[0].columns)

    def submit_answer(self) -> Callable:
        data = self.data.copy()
        if self._goal == "DAG Inference Mission":
            return TaskFactory.create_dag_task(data, is_numeric=self._is_numeric)
        elif self._goal == "Conditional Average Treatment Effect (CATE) Mission":
            return TaskFactory.create_cate_task(data)
        elif self._goal == "Treatment Effect Mission":
            return TaskFactory.create_te_task(data)
        else:
            raise ValueError(f"Unknown process type: {self._goal!r}")
