# Classes
from causalitygame.agents.abstract import BaseAgent
from causalitygame.game_engine.GameInstance import GameInstance

# Science
import numpy as np
import pandas as pd


# Utils
import os
import zlib
import logging
import datetime

# Types
from typing import Any, Callable, Dict, List, Tuple, Optional, TypedDict

# Constants
from causalitygame.lib.constants.environment import (
    OBSERVABLE_COLUMN,
    NODE_PROPERTY_DOMAIN,
    NODE_PROPERTY_MEASURABLE,
    NODE_PROPERTY_TREATABLE,
)
from causalitygame.lib.constants.nodes import (
    ACCESSIBILITY_LATENT,
    ACCESSIBILITY_OBSERVABLE,
    ACCESSIBILITY_CONTROLLABLE,
)


class Hooks(TypedDict, total=False):
    """
    Optional lifecycle hooks for environment behavior introspection or instrumentation.
    """

    on_round_start: Callable[[str, int, Dict, Dict, Dict], None]
    on_action_chosen: Callable[[str, Dict, str, Any], None]
    on_action_evaluated: Callable[[str, Dict, str, Any, Any], None]
    on_round_end: Callable[[str, int, Dict, str, Any, Dict, Any], None]


class Environment:
    """
    The main controller of a causal discovery game simulation.

    It maintains simulation state, interfaces with the agent, validates and executes actions,
    and manages datasets resulting from interventions and observations.

    Attributes:
        game_instance (GameInstance): The SCM and causal game configuration.
        agent (BaseAgent): Agent responsible for decision-making.
        agent_name (str): Identifier for the agent (used in logging and hooks).
        random_state (np.random.RandomState): Global RNG for reproducibility.
        hooks (Hooks): Optional lifecycle hooks.
        current_round (int): Current game round.
        state (Dict): Tracks datasets and final answer.
        history (List[Dict]): Action-state history.
        node_properties (Dict): Treatability, measurability, and domain metadata.
        random_states (Dict): RNGs per hashed treatment key for deterministic interventions.
    """

    def __init__(
        self,
        game_instance: GameInstance,
        agent: BaseAgent,
        agent_name: str,
        random_state: np.random.RandomState,
        hooks: Hooks = {},
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.game_instance = game_instance
        self.agent = agent
        self.agent_name = agent_name
        self.random_state = random_state
        self.hooks = hooks
        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        self.current_round = 0
        self.state = self._initialize_state()
        self.history: List[Dict[str, Any]] = []
        self.node_properties = self._initialize_node_properties()
        self.random_states: Dict[Tuple, Any] = {}

    def _initialize_state(self) -> Dict[str, Any]:
        """
        Initialize the state dictionary with datasets and the final answer.

        Returns:
            Dict[str, Any]: Initial state.
        """
        return {"datasets": {}, "final_answer": None}

    def _initialize_node_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Mark nodes as measurable and treatable with extracted domain.

        Returns:
            Dict[str, Dict[str, Any]]: Metadata per node.
        """
        return {
            node_name: {
                NODE_PROPERTY_TREATABLE: scm_node.accessibility
                == ACCESSIBILITY_CONTROLLABLE,
                NODE_PROPERTY_MEASURABLE: scm_node.accessibility
                != ACCESSIBILITY_LATENT,
                NODE_PROPERTY_DOMAIN: scm_node.domain,
            }
            for node_name, scm_node in sorted(self.game_instance.scm.nodes.items())
        }

    def _hash_treatment(self, treatment: Dict[str, Any]) -> Tuple:
        """
        Converts a dictionary treatment into a hashable sorted tuple key.

        Args:
            treatment (Dict[str, Any]): Intervention dictionary.

        Returns:
            Tuple: Sorted, hashable version of treatment.
        """
        return tuple(sorted(treatment.items()))

    def get_available_actions(self) -> Dict[str, Optional[Any]]:
        """
        Returns the dictionary of all available actions.

        Includes each node (treatment) and special actions like 'observe' and 'stop_with_answer'.

        Returns:
            Dict[str, Optional[Any]]: Action space.
        """
        return {
            node.name: node.domain
            for node in self.game_instance.scm.nodes.values()
            if node.accessibility == ACCESSIBILITY_CONTROLLABLE
        } | {"observe": None, "stop_with_answer": None}

    def get_state(self) -> Dict[str, Any]:
        """
        Snapshot of the environment state passed to the agent.

        Returns:
            Dict[str, Any]: Current datasets, round, action space, and final answer.
        """
        # TODO: Create constants for the columns in the datasets
        return {
            "datasets": self.state["datasets"].copy(),
            "round": self.current_round,
            "available_actions": self.get_available_actions(),
            "final_answer": self.state["final_answer"],
        }

    def apply_action(self, action: str, action_object: Optional[Any] = None) -> None:
        """
        Dispatches execution of the agent's chosen action.

        Args:
            action (str): Action string ('experiment', 'observe', 'stop_with_answer').
            action_object (Any): Parameters required by the action.

        Raises:
            ValueError: If the action or its format is invalid.
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
            return

        if action == "experiment":
            if not isinstance(action_object, list):
                raise ValueError(
                    "Experiment actions must be a list of (dict, int) tuples."
                )
            for treatment, n_samples in action_object:
                if treatment == "observe":

                    self._run_observation(n_samples)
                else:
                    self._run_intervention(treatment, n_samples)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _run_observation(self, num_samples: int) -> None:
        """
        Generates observational (non-interventional) data samples.

        Args:
            num_samples (int): Number of samples to generate.
        """
        samples = self.game_instance.scm.generate_samples(
            num_samples=num_samples, random_state=self.random_state
        )
        if OBSERVABLE_COLUMN not in self.state["datasets"]:
            self.state["datasets"][OBSERVABLE_COLUMN] = pd.DataFrame(
                columns=self.game_instance.scm.nodes.keys()
            )
        # If the observable column already exists, concatenate new samples
        existing_df = self.state["datasets"][OBSERVABLE_COLUMN]
        if not existing_df.empty and not existing_df.isna().all().all():
            self.state["datasets"][OBSERVABLE_COLUMN] = pd.concat(
                [existing_df, samples], ignore_index=True
            )
        else:
            self.state["datasets"][OBSERVABLE_COLUMN] = samples.reset_index(drop=True)

    def _run_intervention(self, treatment: Dict[str, Any], num_samples: int) -> None:
        """
        Executes a single interventional experiment.

        Args:
            treatment (Dict[str, Any]): Intervention mapping.
            num_samples (int): Number of samples to generate.

        Logs:
            Errors for untreatable nodes or invalid values.
        """
        for node, value in treatment.items():
            # Check if the node is treatable
            if (
                node not in self.node_properties
                or not self.node_properties[node][NODE_PROPERTY_TREATABLE]
            ):
                self.logger.error(f"Invalid treatment: node '{node}' is not treatable.")
                return
            domain = self.node_properties[node][NODE_PROPERTY_DOMAIN]
            # Check if the value is in the domain
            if isinstance(value, str) and value not in domain:
                self.logger.error(
                    f"Invalid treatment value '{value}' for node '{node}'"
                )
                return
            elif isinstance(value, (int, float)) and not (
                domain[0] <= value <= domain[1]
            ):
                self.logger.error(
                    f"Treatment value {value} out of domain bounds for node '{node}'"
                )
                return
        # Hash the treatment to use as a key for random state
        hashed = self._hash_treatment(treatment)
        if hashed not in self.random_states:
            # Create a new random state for this treatment
            seed = zlib.crc32(str(hashed).encode())
            rs_base = np.random.RandomState(seed)
            self.random_states[hashed] = (
                self.game_instance.scm.prepare_new_random_state_structure(rs_base)
            )

        # Generate samples using the hashed random state
        samples = self.game_instance.scm.generate_samples(
            interventions=treatment,
            num_samples=num_samples,
            random_state=self.random_states[hashed],
        )
        self.state["datasets"].setdefault(hashed, pd.DataFrame())
        self.state["datasets"][hashed] = pd.concat(
            [self.state["datasets"][hashed], samples], ignore_index=True
        )

    def run_game(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Main game loop. Executes actions chosen by the agent until stopping condition.

        Invokes hooks before/after rounds and actions. Handles lifecycle and logs all history.

        Returns:
            Tuple[Dict[str, Any], pd.DataFrame]: Final state and complete action history.
        """
        while self.current_round <= self.game_instance.max_rounds:
            state = self.get_state()
            # Filter measurable nodes
            measurable_nodes = [
                node.name
                for node in self.game_instance.scm.nodes.values()
                if node.accessibility
                in {ACCESSIBILITY_OBSERVABLE, ACCESSIBILITY_CONTROLLABLE}
            ]
            filtered_samples = {
                k: v[measurable_nodes] for k, v in state["datasets"].items()
            }
            actions = state["available_actions"]

            # Hook: round start
            if hook := self.hooks.get("on_round_start"):
                hook(
                    self.agent_name,
                    self.current_round,
                    state,
                    actions,
                    filtered_samples,
                )

            action, action_object = self.agent.choose_action(
                samples=filtered_samples, actions=actions, num_rounds=self.current_round
            )

            # Hook: action chosen
            if hook := self.hooks.get("on_action_chosen"):
                hook(self.agent_name, state, action, action_object)

            current_result = self.agent.submit_answer()

            # Hook: action evaluated
            if hook := self.hooks.get("on_action_evaluated"):
                hook(self.agent_name, state, action, action_object, current_result)

            # Hook: round end
            if hook := self.hooks.get("on_round_end"):
                hook(
                    self.agent_name,
                    self.current_round,
                    state,
                    action,
                    action_object,
                    filtered_samples,
                    current_result,
                )

            if action == "stop_with_answer":
                break

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

        self.apply_action("stop_with_answer")
        return self.get_state(), pd.DataFrame(self.history)

    def save_game_history(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Save the entire game history as a CSV file.

        Args:
            path (Optional[str]): Optional file path or folder. If not provided, writes to ./output.

        Returns:
            pd.DataFrame: Game history as a DataFrame.
        """
        df = pd.DataFrame(self.history)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_path = path or f"./output/game_history-{timestamp}.csv"
        if not final_path.endswith(".csv"):
            final_path += f"/results-{timestamp}.csv"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        df.to_csv(final_path, index=False)
        return df
