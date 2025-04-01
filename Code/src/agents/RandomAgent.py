import random
from .base import BaseAgent


class RandomAgent(BaseAgent):
    def choose_action(self, samples, actions, num_rounds):
        # 10% chance to stop early
        if "stop_with_answer" in actions and random.random() < 0.1:
            return "stop_with_answer", None

        # Pick a random intervenable node
        node, domain = random.choice(
            [(k, v) for k, v in actions.items() if k != "stop_with_answer"]
        )

        # Generate 1-3 different treatments within the variable's domain
        num_treatments = random.randint(1, 3)
        treatments = []

        for _ in range(num_treatments):
            if isinstance(domain, tuple):  # Numerical domain
                treatment_value = random.uniform(domain[0], domain[1])
            elif isinstance(domain, list):  # Categorical domain
                treatment_value = random.choice(domain)
            else:
                treatment_value = None  # Edge case

            num_samples = random.randint(10, 50)  # Random sample size
            treatments.append(({node: treatment_value}, num_samples))

        return "experiment", treatments

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """
        return "Placeholder_Causal_Model"
