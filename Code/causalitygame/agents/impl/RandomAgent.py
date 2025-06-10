import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ..base import BaseAgent
from causalitygame.lib.scripts.pc import learn as learn_dag
from causalitygame.lib.scripts.empiricalCATE import compute_empirical_cate_fuzzy
from causalitygame.lib.scripts.xgboostTE import te_estimation


class RandomAgent(BaseAgent):
    _is_numeric = False

    def __init__(
        self,
        stop_probability: Optional[float] = None,
        experiments_range: Optional[Tuple[int, int]] = None,
        samples_range: Optional[Tuple[int, int]] = None,
        seed: int = 42,
    ):
        if stop_probability is not None:
            assert (
                0 <= stop_probability <= 1
            ), "stop_probability must be between 0 and 1"
        self.seed = seed
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
        self.past_data: pd.DataFrame = None

    def inform(self, goal: str, behavior_metric: str, deliverable_metric: str):
        """
        Inform the agent about the goal, behavior metric, and deliverable.
        """
        # Store the goal, behavior metric, and deliverable
        self._process = goal.split(":")[0]
        self._goal = goal
        self._behavior_metric = behavior_metric
        self._deliverable_metric = deliverable_metric

    def choose_action(self, samples, actions, num_rounds):
        # Save the samples for future analysis
        self._merge_data(samples)
        # Chance of stopping early
        if (
            "stop_with_answer" in actions
            and self.random_state.random() < self.stop_probability
        ):
            return "stop_with_answer", None

        # Generate 1-3 different treatments within the variable's domain
        num_experiments = self.random_state.randint(
            low=self.experiments_range[0], high=self.experiments_range[1]
        )
        treatments = []

        for _ in range(num_experiments):
            # Pick a random intervenable node
            action_items = [
                (k, v) for k, v in actions.items() if k != "stop_with_answer"
            ]
            node, domain = action_items[self.random_state.randint(len(action_items))]

            # Check for "Observe" action
            if node == "observe":
                # Generate a random sample size
                num_samples = self.random_state.randint(
                    low=self.samples_range[0], high=self.samples_range[1]
                )
                treatments.append(("observe", num_samples))
                continue

            # Check if the domain is categorical
            if type(domain[0]) is str:
                treatment_value = self.random_state.choice(domain)
            else:
                self._is_numeric = True
                treatment_value = self.random_state.uniform(domain[0], domain[1])
            # if isinstance(domain, tuple):  # Numerical domain
            #     self._is_numeric = True
            #     treatment_value = self.random_state.uniform(domain[0], domain[1])
            # elif isinstance(domain, list):  # Categorical domain
            #     treatment_value = self.random_state.choice(domain)
            # else:
            #     treatment_value = None  # Edge case

            num_samples = self.random_state.randint(
                self.samples_range[0], self.samples_range[1]
            )  # Random sample size
            treatments.append(({node: treatment_value}, num_samples))

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
        print(f"Task: {task_mapping.get(self._process, None)}")
        return task_mapping.get(self._process, None)()

    def _dag_inference_task(self):
        return learn_dag(self.past_data.copy(), self._is_numeric, seed=self.seed)

    def _cate_task(self):
        data = self.past_data.copy()

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
