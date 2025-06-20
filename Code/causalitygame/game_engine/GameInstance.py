# Science
import numpy as np

# Utils
import os
import json
from causalitygame.lib.utils.imports import find_importable_classes
from causalitygame.lib.utils.random_state_serialization import (
    random_state_from_json,
    random_state_to_json,
)

# Types
from typing import Dict, Type
from causalitygame.scm.abstract import SCM
from causalitygame.missions.abstract import BaseMission

# Constants
from causalitygame.lib.constants.routes import MISSIONS_FOLDER_PATH


class GameInstance:
    """
    Represents a saved instance of a causality game, encapsulating the SCM, mission, and random state.
    """

    def __init__(
        self,
        max_rounds: int,
        scm: SCM,
        mission: BaseMission,
        random_state: np.random.RandomState,
    ):
        self.max_rounds = max_rounds
        self.scm = scm
        self.mission = mission
        self.random_state = random_state

    def to_dict(self) -> dict:
        return {
            "max_rounds": self.max_rounds,
            "scm": self.scm.to_dict(),
            "mission": self.mission.to_dict(),
            "random_state": random_state_to_json(self.random_state),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameInstance":
        # Identify specific mission classes
        mission_classes: Dict[str, Type[BaseMission]] = find_importable_classes(
            MISSIONS_FOLDER_PATH, base_class=BaseMission
        )
        # Check if the mission class is known
        mission_cls = mission_classes.get(data["mission"]["class"])
        if mission_cls is None:
            raise ValueError(f"Unknown mission class: {data['mission']['class']}")
        # Instantiate the mission from the data
        mission = mission_cls.from_dict(data["mission"])
        # Generate the SCM
        scm = SCM.from_dict(data["scm"])
        # Generate the random state
        random_state = random_state_from_json(data["random_state"])
        return cls(data["max_rounds"], scm, mission, random_state)

    def save(self, filename: str) -> None:
        """
        Serialize the game instance to disk as a JSON file.
        """
        # Get the directory from the filename
        directory = os.path.dirname(filename)
        # Check if the directory exists, if not, create it
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Write the JSON file
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filename: str) -> "GameInstance":
        """
        Load a game instance from a JSON file.
        """
        # Check if the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"GameInstance file not found: {filename}")
        # Read the JSON file
        with open(filename, "r") as f:
            data = json.load(f)
        # Deserialize the game instance from the data
        return cls.from_dict(data)
