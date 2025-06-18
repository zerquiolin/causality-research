# Logging
import logging

# Math
import numpy as np

# Data
import pandas as pd
from causalitygame.scm.nodes.base import (
    ACCESSIBILITY_OBSERVABLE,
    ACCESSIBILITY_CONTROLLABLE,
)

# Models
from causalitygame.game_engine.GameInstance import GameInstance
from causalitygame.agents.base import BaseAgent

# Typing
from typing import Any, Callable, Dict, List, Tuple, Optional, TypedDict


class Hooks(TypedDict):
    """
    Hooks for the game.
    """

    on_round_start: Optional[Callable[[str, int, Dict, List, Dict], None]] = None
    on_action_chosen: Optional[Callable[[str, Dict, str, any], None]] = None
    on_action_evaluated: Optional[Callable[[str, Dict, str, any, Tuple], None]] = None
    on_round_end: Optional[Callable[[str, int, Dict, str, any, Dict, Tuple], None]] = (
        None
    )


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
        agent_name: str,
        random_state: np.random.RandomState,
        hooks: Hooks = {},
        logger: logging.Logger = None,
    ) -> None:
        """
        Initializes the Environment.

        Args:
            game_instance (GameInstance): The game instance containing the SCM.
            agent (BaseAgent): The agent controlling interventions.
            random_state (np.random.RandomState): Random State for reproducibility.
        """
        self.random_state: np.random.Generator = random_state
        self.game_instance: GameInstance = game_instance
        self.agent: BaseAgent = agent
        self.agent_name: str = agent_name
        self.current_round: int = 0
        self.state: Dict[str, Any] = self.initialize_state()
        self.history: List[Dict[str, Any]] = (
            []
        )  # Stores game history: round, state, action, action_object
        self.node_properties: Dict[str, Dict[str, Any]] = (
            self.initialize_node_properties()
        )
        self.hooks: Hooks = hooks
        self.random_states: Dict[Any, np.random.RandomState] = {}
        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        )

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
        #  TODO: Fix all the logic regarding node properties
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
            node.name: node.domain for node in self.game_instance.scm.nodes.values()
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
            self.history.append(
                {
                    "round": self.current_round,
                    "action": "final_answer",
                    "action_object": answer,
                    "current_result": answer,
                    "state_datasets": self.state["datasets"],
                }
            )
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
                    self.logger.error("Error: Invalid experiment. Node not treatable.")
                    return
                # Validate that the treatment values are within the node's domain
                if not all(
                    (
                        (
                            value >= self.node_properties[node]["domain"][0]
                            and value <= self.node_properties[node]["domain"][1]
                        )
                        if type(value) is not str
                        else value in self.node_properties[node]["domain"]
                    )
                    for node, value in experiment[0].items()
                ):
                    self.logger.error("Error: Invalid experiment. Value not in domain.")
                    return

                self.perform_experiment([experiment])
        else:
            print(f"Invalid action: {action}")

    def perform_experiment(self, treatments: List[Tuple[Any, int]]) -> None:
        """
        Executes a batch of intervention experiments.

        For each treatment, generates samples using the SCM and stores them under the treatment key.

        Args:
            treatments (List[Tuple[Any, int]]): A list of (treatment, num_samples) pairs.
        """
        self.logger.info(
            "Starting experiment with the following treatments: %s", treatments
        )
        for treatment, num_samples in treatments:
            if treatment == "observe":
                samples = self.game_instance.scm.generate_samples(
                    num_samples=num_samples, random_state=self.random_state
                )
                if "empty" not in self.state["datasets"]:
                    self.logger.info(
                        "No empty dataset found. Creating an empty dataset for observation."
                    )
                    self.state["datasets"]["empty"] = pd.DataFrame(
                        columns=[node for node in self.game_instance.scm.nodes.keys()]
                    )
                self.state["datasets"]["empty"] = pd.concat(
                    [self.state["datasets"]["empty"], samples], ignore_index=True
                )
                continue

            # Generate a hashable representation of the treatment to use a dedicated random state
            hashable_treatment = tuple(sorted(treatment.items()))
            if hashable_treatment not in self.random_states:
                seed = zlib.crc32(str(hashable_treatment).encode())
                self.logger.info(
                    "Creating new random states for all variables under treatment %s with seed %s",
                    hashable_treatment,
                    seed,
                )
                rs_base = np.random.RandomState(seed)
                # TODO: Check if this is still needed
                # self.random_states[hashable_treatment] = rs_base
                self.random_states[hashable_treatment] = (
                    self.game_instance.scm.prepare_new_random_state_structure(rs_base)
                )

            self.logger.debug("Generating %s samples.", num_samples)
            assert isinstance(
                self.random_states[hashable_treatment],
                dict,
            ), (
                "Random state for treatment must be a numpy RandomState instance, "
                f"got {type(self.random_states[hashable_treatment])}"
            )
            samples = self.game_instance.scm.generate_samples(
                interventions=treatment,
                num_samples=num_samples,
                random_state=self.random_states[hashable_treatment],
            )
            self.logger.debug(
                "Done. Now incorporating the samples into the state. Drawn samples: \n\n%s",
                samples,
            )

            if hashable_treatment not in self.state["datasets"]:
                self.state["datasets"][hashable_treatment] = samples
            else:
                self.state["datasets"][hashable_treatment] = pd.concat(
                    [self.state["datasets"][hashable_treatment], samples],
                    ignore_index=True,
                )
            self.logger.debug("Done.")

    def run_game(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Runs the simulation until the agent stops or the maximum number of rounds is reached.

        At each round, the agent is provided with the current state and available actions,
        chooses an action, and the action is applied.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: The final state and the history of state-action pairs.
        """
        while self.current_round < self.game_instance.max_rounds:
            state = self.get_state()
            samples = state["datasets"]
            for treatment, dataset in samples.items():
                # Filter columns to only those that are measurable
                measurable_columns = [
                    node.name
                    for node in self.game_instance.scm.nodes.values()
                    if node.accessibility == ACCESSIBILITY_OBSERVABLE
                    or node.accessibility == ACCESSIBILITY_CONTROLLABLE
                ]
                samples[treatment] = dataset[measurable_columns]
            actions = state["available_actions"]
            num_rounds = state["round"]

            # Hook for round start
            if "on_round_start" in self.hooks and callable(
                self.hooks["on_round_start"]
            ):
                self.hooks["on_round_start"](
                    self.agent_name, num_rounds, state, actions, samples
                )

            action, action_object = self.agent.choose_action(
                samples=samples, actions=actions, num_rounds=num_rounds
            )

            # Hook for action chosen
            if "on_action_chosen" in self.hooks and callable(
                self.hooks["on_action_chosen"]
            ):
                self.hooks["on_action_chosen"](
                    self.agent_name, state, action, action_object
                )

            # For performance measurement purposes only
            current_result = self.agent.submit_answer()

            # Hook for action applied
            if "on_action_evaluated" in self.hooks and callable(
                self.hooks["on_action_evaluated"]
            ):
                self.hooks["on_action_evaluated"](
                    self.agent_name, state, action, action_object, current_result
                )

            # Hook for round end
            if "on_round_end" in self.hooks and callable(self.hooks["on_round_end"]):
                self.hooks["on_round_end"](
                    self.agent_name,
                    num_rounds,
                    state,
                    action,
                    action_object,
                    samples,
                    current_result,
                )

            # Check if the agent has chosen to stop the game
            if action == "stop_with_answer":
                break

            # Log the current state and action
            self.history.append(
                {
                    "round": self.current_round,
                    "action": action,
                    "action_object": action_object,
                    "current_result": current_result,
                    "state_datasets": state["datasets"],
                }
            )

            self.apply_action(action, action_object)
            self.current_round += 1

        # Game termination
        self.apply_action("stop_with_answer")

        return self.get_state(), pd.DataFrame(self.history)

    def save_game_history(self, path: str = None) -> pd.DataFrame:
        """
        Converts the stored state-action history into a Pandas DataFrame and saves it as a CSV file.

        Args:
            path (str): The path where the CSV file will be saved.

        Returns:
            pd.DataFrame: The DataFrame containing the game history.
        """
        history_df = pd.DataFrame(self.history)
        if path and not path.endswith(".csv"):
            if not path.endswith("/"):
                path += "/"
            path += (
                f"results-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            )
        file_path = (
            path
            or f"./output/game_history-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        history_df.to_csv(file_path, index=False)
        return history_df
