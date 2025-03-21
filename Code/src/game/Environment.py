import os
import datetime
import pandas as pd
import numpy as np
import zlib


class Environment:
    def __init__(
        self, game_instance, agent, random_state: np.random.Generator, max_rounds=10
    ):
        """
        Initialize the game environment.

        :param game_instance: The GameInstance object containing the SCM and initial state.
        :param agent: The agent (player) controlling interventions.
        :param max_rounds: Maximum rounds before forced termination.
        """
        self.random_state = random_state
        self.game_instance = game_instance
        self.agent = agent
        self.max_rounds = max_rounds
        self.current_round = 0
        self.state = self.initialize_state()
        self.history = []  # Stores (round, state, action, action_object)
        self.node_properties = self.initialize_node_properties()
        self.random_states = {}

    def initialize_state(self):
        """
        Initializes the environment state.

        - The state contains datasets structured as:
          { node: { value: dataset, value2: dataset } }
        - Nodes with no samples are marked as `"empty"`.
        """
        return {"datasets": {}, "final_answer": None}

    def initialize_node_properties(self):
        """
        Identifies whether each node is:
        - Treatable (intervenable)
        - Measurable (observable)
        """
        properties = {}
        for node, scm_node in sorted(
            self.game_instance.scm.nodes.items(), key=lambda x: x[0]
        ):
            properties[node] = {
                "treatable": self.random_state.choice([True, False]),
                "measurable": self.random_state.choice([True, False]),
                "domain": scm_node.domain,  # Extract domain directly from SCMNode
            }
        return properties

    def get_available_actions(self):
        """
        Returns a dictionary where:
        - Keys: Treatable nodes (or "stop_with_answer").
        - Values: Domains of the variables.
        """
        actions = {
            node: props["domain"]
            for node, props in self.node_properties.items()
            if props["treatable"]
        }
        actions["stop_with_answer"] = None  # Stopping condition
        return actions

    def get_state(self):
        """
        Returns the current game state.
        """
        return {
            "datasets": self.state["datasets"],
            "round": self.current_round,
            "available_actions": self.get_available_actions(),
        }

    def apply_action(self, action, action_object=None):
        """
        Applies the chosen action.

        - If **"experiment"**, executes interventions using `action_object` (list of (treatment_dict, num_samples)).
        - If **"stop_with_answer"**, stores agent's answer and ends the game.
        """
        if action == "stop_with_answer":
            answer = self.agent.submit_answer()
            self.state["final_answer"] = answer
            print(f"Game Over: Agent submitted answer {answer}.")
        elif (
            action in self.node_properties and self.node_properties[action]["treatable"]
        ):
            if not isinstance(action_object, list) or not all(
                isinstance(t, tuple) for t in action_object
            ):
                print(
                    "Error: Invalid experiment format. Expected [(treatment_dict, num_samples), ...]"
                )
                return

            self.perform_experiment(action_object)
        else:
            print(f"Invalid action: {action}")

    def perform_experiment(self, treatments):
        """
        Executes a batch of intervention experiments.

        :param treatments: List of (treatment_dict, num_samples)
        """
        for treatment, num_samples in treatments:
            # Hashable treatment
            hashable_treatment = tuple(sorted(treatment.items()))
            if hashable_treatment not in self.random_states:
                self.random_states[hashable_treatment] = np.random.RandomState(
                    zlib.crc32(str(hashable_treatment).encode())
                )

            samples = self.game_instance.scm.generate_samples(
                interventions=treatment,
                num_samples=num_samples,
                random_state=self.random_states[hashable_treatment],
            )
            for node, value in treatment.items():
                if node not in self.state["datasets"]:
                    self.state["datasets"][node] = {}

                # Store dataset under the specific treatment value
                self.state["datasets"][node][value] = samples

    def run_game(self):
        """
        Runs the simulation until:
        - The agent submits an answer (`stop_with_answer`).
        - The maximum number of rounds is reached.
        """
        while self.current_round < self.max_rounds:
            state = self.get_state()
            action, action_object = self.agent.choose_action(state)
            print(
                f"Round {self.current_round}: Agent chose action '{action}' with object: {action_object}"
            )

            # Store the (state, action) pair in history
            self.history.append(
                {
                    "round": self.current_round,
                    "action": action,
                    "action_object": action_object,
                    "state_datasets": state["datasets"],
                }
            )

            if action == "stop_with_answer":
                self.apply_action(action)
                break  # End the game

            self.apply_action(action, action_object)
            self.current_round += 1

        print("Game ended.")
        return self.get_state()

    def get_game_history(self):
        """
        Convert the stored state-action history into a Pandas DataFrame.
        """
        history = pd.DataFrame(self.history)

        # File Path
        file_path = f"./output/game_history-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        # Extract the directory from the file path
        directory = os.path.dirname(file_path)

        # Ensure the directory exists
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write the CSV file
        history.to_csv(file_path, index=False)

        return history
