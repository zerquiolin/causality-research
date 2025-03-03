from src.lib.classes.abstract.MDP import MDP
from src.lib.classes.network.SCM import StructuralCausalModel


class Environment(MDP):
    def __init__(self, bayesian_network: StructuralCausalModel):
        self.bayesian_network = bayesian_network
        self.interventions = {node: [0, 1] for node in bayesian_network.nodes}
        self.state = dict()

    def is_terminal(self):
        """
        Check if there are no more interventions left to perform (terminal state).
        """
        return len(self.interventions) == 0

    def step(self, action, value, return_object=False):
        """
        Apply a binary intervention (0, 1) to the environment, updating the state.

        Args:
            action (str): The variable to intervene on.
            value (int): The binary value (0, 1) for the intervention.
            return_object (bool, optional): Whether to return the updated object or not. Defaults to False.

        Returns:
            Environment or None: The updated Environment object if return_object is True, otherwise None.

        Raises:
            ValueError: If the action is not valid or the value is not valid for the given action.
        """
        # Check Action
        if action not in self.interventions:
            raise ValueError(f"Action {action} is not valid.")

        # Check Value
        if value not in self.interventions[action]:
            raise ValueError(
                f"Value {value} is not valid. Must be {self.interventions[action]}."
            )

        # Create the intervention
        intervention = {action: value}

        # Infer the probabilities
        inference = self.bayesian_network.infer_probability(interventions=intervention)

        # Update the state
        if action not in self.state:
            self.state[action] = dict()

        self.state[action][value] = inference

        # Remove the action from the available interventions
        if len(self.interventions[action]) == 1:
            del self.interventions[action]
        else:
            self.interventions[action].remove(value)

        return self if return_object else None

    def available_actions(self):
        """
        Return the list of available actions (interventions) left in the environment.
        """
        return self.interventions

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.state = {node: [0, 1] for node in self.bayesian_network.nodes}
        print("Environment has been reset.")


if __name__ == "__main__":
    from models.Covid import Covid

    # Create a Bayesian Network
    covid = Covid().gen_model()

    # Create an environment
    env = Environment(covid)

    # Actions
    print(env.available_actions())
