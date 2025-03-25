import random
from src.lib.models.abstract.BaseAgent import BaseAgent


class GreedyAgent(BaseAgent):
    _is_first_round = True

    def choose_action(self, samples, actions, num_rounds):
        # Check if this is the first round
        if self._is_first_round:
            self._is_first_round = False
        else:
            return "stop_with_answer", None

        # Define the treatment list
        treatments = []
        # Iterate over all possible actions
        for node in actions.keys():
            print(f"node: {node}")
            # Skip the stop_with_answer action
            if node == "stop_with_answer":
                continue
            # Get the domain of the action
            domain = actions[node]
            print(f"domain: {domain}")
            # Check if the domain is categorical
            if isinstance(domain[0], str):
                # If the domain is categorical, choose a random value
                treatments.append(
                    ({node: random.choice(domain)}, random.randint(10, 50))
                )
            else:
                # If the domain is numerical, choose a random value between the lower and upper bounds
                treatments.append(
                    (
                        {node: random.uniform(domain[0], domain[1])},
                        random.randint(10, 50),
                    )
                )

        return "play", treatments

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """
        return "Placeholder_Causal_Model"
