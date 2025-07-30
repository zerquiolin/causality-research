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
from typing import Callable, Dict, Optional, Tuple


class RandomAgent(BaseAgent):
    """
    An random agent that selects random interventions or observations,
    and then computes the requested causal or treatment‐effect task.
    """

    _is_numeric = False

    def __init__(
        self,
        stop_probability: Optional[float] = None,
        experiments_range: Optional[Tuple[int, int]] = None,
        samples_range: Optional[Tuple[int, int]] = None,
        seed: int = 42,
    ):
        # Checks
        if stop_probability is not None:
            assert (
                0 <= stop_probability <= 1
            ), "stop_probability must be between 0 and 1"
        # Initialize the base class
        super().__init__()
        # Initialize the agent's parameters
        self.seed = seed
        self._is_numeric = False
        self.data: pd.DataFrame = None
        self.random_state = np.random.RandomState(seed)
        self.stop_probability = (
            stop_probability if stop_probability else self.random_state.beta(0.5, 10)
        )
        self.experiments_range = (
            experiments_range
            if experiments_range
            else (1, max(self.random_state.poisson(10), 2))
        )
        self.samples_range = (
            samples_range
            if samples_range
            else (
                self.random_state.randint(10, 50),
                self.random_state.randint(50, 100),
            )
        )

    def inform(self, goal: str, behavior_metric: str, deliverable_metric: str):
        """
        Inform the agent about the goal, behavior metric, and deliverable.
        """
        # Store the goal, behavior metric, and deliverable
        self._goal = goal["goal"]
        self._behavior_metric = behavior_metric
        self._deliverable_metric = deliverable_metric

    def choose_action(self, samples, actions, num_rounds):
        # Save the samples for future analysis
        self._update_data(samples)

        # Chance of stopping early
        # if (
        #     STOP_WITH_ANSWER_ACTION in actions
        #     and self.random_state.random() < self.stop_probability
        # ):
        #     return STOP_WITH_ANSWER_ACTION, None

        # Generate different treatments within the variable's domain
        num_experiments = self.random_state.randint(
            low=self.experiments_range[0], high=self.experiments_range[1]
        )

        # Define the treatment list
        treatments = []
        for _ in range(num_experiments):
            # Pick a random intervenable node
            action_items = [
                (k, v) for k, v in actions.items() if k != STOP_WITH_ANSWER_ACTION
            ]
            node, domain = action_items[self.random_state.randint(len(action_items))]

            # Check for "Observe" action
            if node == OBSERVE_ACTION:
                # Generate a random sample size
                num_samples = self.random_state.randint(
                    low=self.samples_range[0], high=self.samples_range[1]
                )
                treatments.append((OBSERVE_ACTION, num_samples))
                continue

            # Check domain type
            if type(domain[0]) is str:  # Categorical domain
                treatment_value = self.random_state.choice(domain)
            else:  # Numerical domain
                self._is_numeric = True
                if type(domain) is list:  # Discrete numerical domain
                    treatment_value = self.random_state.choice(domain)
                else:  # Continuous numerical domain
                    treatment_value = self.random_state.uniform(domain[0], domain[1])

            num_samples = self.random_state.randint(
                self.samples_range[0], self.samples_range[1]
            )  # Random sample size
            treatments.append(({node: treatment_value}, num_samples))

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
        # print(f"Submitting answer with data:\n{len(data)}")
        if self._goal == "DAG Inference Mission":
            return TaskFactory.create_dag_task(data, is_numeric=self._is_numeric)
        elif self._goal == "Treatment Effect Mission":
            return TaskFactory.create_te_task(data)
        elif self._goal == "Average Treatment Effect Mission":
            return TaskFactory.create_ate_task(data)
        elif self._goal == "Conditional Average Treatment Effect Mission":
            return TaskFactory.create_cate_task(data)
        else:
            raise ValueError(f"Unknown process type: {self._goal!r}")
