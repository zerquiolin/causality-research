import numpy as np
from typing import Any, Dict, List, Tuple
from ..base import BaseAgent
from causalitygame.lib.scripts.pc import learn as learn_dag


class RandomAgent(BaseAgent):
    def __init__(
        self,
        stop_probability: float,
        experiments_range: Tuple[int, int] = (1, 5),
        samples_range: Tuple[int, int] = (100, 1000),
        seed: int = 42,
    ):
        assert 0 <= stop_probability <= 1, "stop_probability must be between 0 and 1"
        self.stop_probability = stop_probability
        self.experiments_range = experiments_range
        self.samples_range = samples_range
        self.random_state = np.random.RandomState(seed)
        self.seed = seed
        self.past_data: Dict[str, Any] = {
            "empty": [],
        }

    def inform(self, goal: str, behavior_metric: str, deliverable_metric: str):
        """
        Inform the agent about the goal, behavior metric, and deliverable.
        """
        # Store the goal, behavior metric, and deliverable
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

            if isinstance(domain, tuple):  # Numerical domain
                treatment_value = self.random_state.uniform(domain[0], domain[1])
            elif isinstance(domain, list):  # Categorical domain
                treatment_value = self.random_state.choice(domain)
            else:
                treatment_value = None  # Edge case

            num_samples = self.random_state.randint(10, 50)  # Random sample size
            treatments.append(({node: treatment_value}, num_samples))

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
        return learn_dag(self.past_data, seed=self.seed)
