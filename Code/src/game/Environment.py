# Logging
import logging

# Math
import numpy as np

# Data
import pandas as pd

# Models
from src.game.GameInstance import GameInstance
from src.agents.base import BaseAgent

# Typing
from typing import Any, Dict, List, Tuple, Optional

# Utils
import os
import zlib
import datetime


class Environment:
    """
    Represents the simulation environment for running causal discovery experiments.

    This environment encapsulates the game instance, the agent controlling interventions,
    and the state of the simulation. It manages experiment execution, state updates, and
    game history logging.

    Attributes:
        game_instance (GameInstance): The instance containing the SCM and initial configuration.
        agent (BaseAgent): The agent that makes intervention decisions.
        random_state (np.random.RandomState): Random State for reproducibility.
        max_rounds (int): Maximum number of rounds before termination.
        current_round (int): The current round number.
        state (Dict): Dictionary holding the current datasets and final answer.
        history (List): List of dictionaries logging (round, state, action, action_object).
        node_properties (Dict): Contains properties (treatable, measurable, domain) for each node.
        random_states (Dict): Mapping of treatments to their dedicated random states.
    """

    def __init__(
        self,
        game_instance: GameInstance,
        agent: BaseAgent,
        random_state: np.random.RandomState,
        max_rounds: int = 10,
    ) -> None:
        """
        Initializes the Environment.

        Args:
            game_instance (GameInstance): The game instance containing the SCM.
            agent (BaseAgent): The agent controlling interventions.
            random_state (np.random.RandomState): Random State for reproducibility.
            max_rounds (int, optional): Maximum rounds before forced termination. Defaults to 10.
        """
        self.random_state: np.random.Generator = random_state
        self.game_instance: GameInstance = game_instance
        self.agent: BaseAgent = agent
        self.max_rounds: int = max_rounds
        self.current_round: int = 0
        self.state: Dict[str, Any] = self.initialize_state()
        self.history: List[Dict[str, Any]] = (
            []
        )  # Stores game history: round, state, action, action_object
        self.node_properties: Dict[str, Dict[str, Any]] = (
            self.initialize_node_properties()
        )
        self.random_states: Dict[Any, np.random.RandomState] = {}

    def initialize_state(self) -> Dict[str, Any]:
        """
        Initializes the simulation state.

        The state includes:
          - 'datasets': A dictionary to hold datasets per node and treatment.
          - 'final_answer': The answer provided by the agent when stopping the game.

        Returns:
            Dict[str, Any]: The initial state.
        """
        return {"datasets": {}, "final_answer": None}

    def initialize_node_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Initializes properties for each node from the SCM.

        Each node is randomly marked as treatable and measurable.
        The node's domain is extracted directly from its SCM node.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping node names to their properties.
        """
        # todo: Add a parameter that allows to have non treatable and non measurable nodes
        properties: Dict[str, Dict[str, Any]] = {}
        # Sort nodes by name (assuming names like 'X1', 'X2', ...)
        for node, scm_node in sorted(
            self.game_instance.scm.nodes.items(), key=lambda x: x[0]
        ):
            properties[node] = {
                # "treatable": self.random_state.choice([True, False]),
                # todo: return this to normal
                "treatable": True,
                "measurable": self.random_state.choice([True, False]),
                "domain": scm_node.domain,
            }
        return properties

    def get_available_actions(self) -> Dict[str, Optional[Any]]:
        """
        Determines available actions for the agent.

        Available actions include:
          - Treatable nodes mapped to their domains.
          - Special actions: "observe" and "stop_with_answer".

        Returns:
            Dict[str, Optional[Any]]: A dictionary of action names to domain or None.
        """
        actions = {
            node: props["domain"]
            for node, props in self.node_properties.items()
            if props["treatable"]
        }
        actions["observe"] = None
        actions["stop_with_answer"] = None
        return actions

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the current game state.

        Returns:
            Dict[str, Any]: A dictionary containing datasets, current round, available actions, and final answer.
        """
        return {
            "datasets": self.state["datasets"],
            "round": self.current_round,
            "available_actions": self.get_available_actions(),
            "final_answer": self.state["final_answer"],
        }

    def apply_action(self, action: str, action_object: Optional[Any] = None) -> None:
        """
        Applies the specified action.

        For "stop_with_answer", the agent's answer is submitted.
        For "experiment", the experiment(s) provided in action_object are executed.
        Invalid actions are logged.

        Args:
            action (str): The action to apply.
            action_object (Optional[Any], optional): Additional parameters for the action.
        """
        if action == "stop_with_answer":
            answer = self.agent.submit_answer()
            self.state["final_answer"] = answer
        elif action == "experiment":
            for experiment in action_object:
                # Handle observe action separately
                if experiment[0] == "observe":
                    self.perform_experiment([experiment])
                    continue
                # Validate that the experiment's nodes are treatable
                if not all(
                    node in self.node_properties
                    and self.node_properties[node]["treatable"]
                    for node in experiment[0].keys()
                ):
                    logging.error("Error: Invalid experiment. Node not treatable.")
                    return
                # Validate that the treatment values are within the node's domain
                if not all(
                    value in self.node_properties[node]["domain"]
                    for node, value in experiment[0].items()
                ):
                    logging.error("Error: Invalid experiment. Value not in domain.")
                    return

                self.perform_experiment([experiment])
        else:
            error(f"Invalid action: {action}")

    def perform_experiment(self, treatments: List[Tuple[Any, int]]) -> None:
        """
        Executes a batch of intervention experiments.

        For each treatment, generates samples using the SCM and stores them under the treatment key.

        Args:
            treatments (List[Tuple[Any, int]]): A list of (treatment, num_samples) pairs.
        """
        for treatment, num_samples in treatments:
            if treatment == "observe":
                samples = self.game_instance.scm.generate_samples(
                    num_samples=num_samples, random_state=self.random_state
                )
                self.state["datasets"].setdefault("empty", []).extend(samples)
                continue

            # Generate a hashable representation of the treatment to use a dedicated random state
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
                self.state["datasets"].setdefault(node, {}).setdefault(
                    value, []
                ).extend(
                    samples
                )  # todo: check if this is correct

    def run_game(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Runs the simulation until the agent stops or the maximum number of rounds is reached.

        At each round, the agent is provided with the current state and available actions,
        chooses an action, and the action is applied.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: The final state and the history of state-action pairs.
        """
        while self.current_round < self.max_rounds:
            state = self.get_state()
            samples = state["datasets"]
            # todo: filter samples to show only measurable nodes
            actions = state["available_actions"]
            num_rounds = state["round"]

            action, action_object = self.agent.choose_action(
                samples=samples, actions=actions, num_rounds=num_rounds
            )

            # Log the current state and action
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
                break

            self.apply_action(action, action_object)
            self.current_round += 1

        return self.get_state(), self.history

    def get_game_history(self) -> pd.DataFrame:
        """
        Converts the stored state-action history into a Pandas DataFrame and saves it as a CSV file.

        Returns:
            pd.DataFrame: The DataFrame containing the game history.
        """
        history_df = pd.DataFrame(self.history)
        file_path = f"./output/game_history-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        history_df.to_csv(file_path, index=False)
        return history_df
