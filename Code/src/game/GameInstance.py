import os
import json
import numpy as np
from networkx.readwrite import json_graph

from src.generators.dag_generator import DAGGenerator
from src.scm.dag import DAG
from src.generators.scm_generator import SCMGenerator
from src.scm.scm import SCM


class GameInstance:
    def __init__(self, scm: SCM, random_state):
        self.scm = scm
        self.random_state = random_state

    def to_dict(self):
        # Use the 'edges' kwarg to address the FutureWarning.
        scm_data = self.scm.to_dict()
        # Convert the state to a JSON-friendly format
        state_dict = {
            "state": self.random_state.get_state()[0],  # 'MT19937'
            "keys": self.random_state.get_state()[
                1
            ].tolist(),  # Convert NumPy array to list
            "pos": self.random_state.get_state()[2],
            "has_gauss": self.random_state.get_state()[3],
            "cached_gaussian": self.random_state.get_state()[4],
        }
        return {"scm": scm_data, "random_state": state_dict}

    @classmethod
    def from_dict(cls, data):
        random_state_config = (
            str(data["random_state"]["state"]),  # Ensure it's a string ('MT19937')
            np.array(
                data["random_state"]["keys"], dtype=np.uint32
            ),  # Ensure NumPy array
            int(data["random_state"]["pos"]),  # Ensure integer
            int(data["random_state"]["has_gauss"]),  # Ensure integer (0 or 1)
            float(data["random_state"]["cached_gaussian"]),  # Ensure float
        )
        random_state = np.random.RandomState()
        random_state.set_state(random_state_config)
        scm = SCM.from_dict(data["scm"])
        return cls(scm, random_state)

    def save(self, filename):
        """Ensure the directory exists and save the game instance as a JSON file."""
        # Extract the directory from the file path
        directory = os.path.dirname(filename)

        # Ensure the directory exists
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write the JSON file
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename):
        """Load a game instance from a JSON file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Write the JSON file
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class GameInstanceCreator:
    """
    Creates a game instance by first generating a DAG using the DAGGenerator
    and then generating an SCM using the SCMGenerator.
    """

    def __init__(
        self,
        dag_generator_params: dict,
        scm_generator_params: dict,
        random_state_seed=911,
    ):
        self.dag_generator_params = dag_generator_params
        self.scm_generator_params = scm_generator_params
        self.random_state_seed = random_state_seed

    def create_instance(self) -> GameInstance:
        # Random state for reproducibility.
        random_state = np.random.RandomState(self.random_state_seed)

        # DAG generation.
        dag_gen = DAGGenerator(**self.dag_generator_params, random_state=random_state)
        dag = dag_gen.generate()

        # SCM generation.
        scm_gen = SCMGenerator(
            dag=dag, **self.scm_generator_params, random_state=random_state
        )
        scm = scm_gen.generate()

        # Return the game instance.
        return GameInstance(scm, random_state=random_state)
