# File: causality_game/agents/random_agent.py
import random
from src.lib.models.abstract.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    def choose_action(self, state):
        """
        Randomly selects between stopping or experimenting.

        - If stopping, returns ("stop_with_answer", None).
        - If experimenting, returns (node, [(treatment_dict, num_samples)]).
        """
        available_actions = state["available_actions"]

        # 20% chance to stop early
        if "stop_with_answer" in available_actions and random.random() < 0.1:
            return "stop_with_answer", None

        # Pick a random intervenable node
        node, domain = random.choice(
            [(k, v) for k, v in available_actions.items() if k != "stop_with_answer"]
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

        return node, treatments

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """
        return "Placeholder_Causal_Model"
